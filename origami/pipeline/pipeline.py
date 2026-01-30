"""ORIGAMI Pipeline - end-to-end training and inference.

Provides a unified API for training ORIGAMI models on JSON data and
performing inference with automatic preprocessing and inverse transforms.
"""

from __future__ import annotations

import cProfile
import pstats
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from origami.config import DataConfig, ModelConfig, OrigamiConfig, TrainingConfig
from origami.inference import OrigamiEmbedder, OrigamiEvaluator, OrigamiGenerator, OrigamiPredictor
from origami.model import OrigamiModel
from origami.preprocessing import NumericDiscretizer, NumericScaler
from origami.tokenizer import JSONTokenizer
from origami.training.metrics import MetricSpec
from origami.utils.device import auto_device

if TYPE_CHECKING:
    from origami.training import TrainResult


class OrigamiPipeline:
    """End-to-end ORIGAMI pipeline for training and inference.

    Combines preprocessing, tokenization, model training, and inference
    into a single unified API. Handles all the complexity of numeric
    scaling/discretization, vocabulary management, and checkpoint saving.

    Example - Training:
        ```python
        from origami import OrigamiPipeline, OrigamiConfig, ModelConfig, DataConfig

        # Minimal - just works with defaults
        pipeline = OrigamiPipeline()
        pipeline.fit(train_data, epochs=20)
        pipeline.save("model.pt")

        # With continuous head for high-cardinality numerics
        config = OrigamiConfig(
            model=ModelConfig(d_model=128),
            data=DataConfig(numeric_mode="scale"),
        )
        pipeline = OrigamiPipeline(config)
        pipeline.fit(train_data, eval_data=eval_data, epochs=50)
        ```

    Example - Inference:
        ```python
        # Load trained pipeline
        pipeline = OrigamiPipeline.load("model.pt")

        # Predict values (returns original scale, not scaled)
        prediction = pipeline.predict({"a": 3.5}, target_key="b")

        # Generate complete objects
        samples = pipeline.generate(num_samples=10)

        # Get embeddings
        embedding = pipeline.embed({"a": 3.5, "b": 42.7})
        ```

    Attributes:
        config: Pipeline configuration
        model: Underlying ORIGAMI model (available after fit/load)
        tokenizer: JSON tokenizer (available after fit/load)
    """

    def __init__(self, config: OrigamiConfig | None = None):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration. Uses sensible defaults if None.
        """
        self.config = config or OrigamiConfig()

        # Internal state - set during fit() or load()
        self._preprocessor: NumericScaler | NumericDiscretizer | None = None
        self._tokenizer: JSONTokenizer | None = None
        self._model: OrigamiModel | None = None
        self._schema: dict | None = None
        self._fitted = False
        self._train_result: TrainResult | None = None

        # Stored preprocessed data for train()
        self._train_processed: list[dict] | None = None
        self._eval_processed: list[dict] | None = None

        # Training state for checkpoint resumption
        # Populated by load() when loading a checkpoint with training state
        self._training_state: dict | None = None

        # Device management
        # _training_device: resolved device for training (GPU/MPS if available)
        # After inference, model stays on CPU (faster for autoregressive generation)
        self._training_device: torch.device | None = None

        # Lazy-initialized inference components
        self._generator: OrigamiGenerator | None = None
        self._predictor: OrigamiPredictor | None = None
        self._embedder: OrigamiEmbedder | None = None

    @property
    def model(self) -> OrigamiModel | None:
        """Get the underlying model (None before fit/load)."""
        return self._model

    @property
    def tokenizer(self) -> JSONTokenizer | None:
        """Get the tokenizer (None before fit/load)."""
        return self._tokenizer

    @property
    def schema(self) -> dict | None:
        """Get the current JSON Schema (None if not set).

        The schema is populated after preprocess() when infer_schema=True
        or when a schema is provided in DataConfig. It can also be set
        manually after loading a model.
        """
        return self._schema

    @schema.setter
    def schema(self, value: dict | None) -> None:
        """Set or replace the JSON Schema.

        Invalidates cached inference components (generator/predictor)
        since they depend on schema for constraint application.
        """
        self._schema = value
        self._generator = None
        self._predictor = None

    def _resolve_device(self) -> torch.device:
        """Resolve the configured device string to an actual device.

        Returns:
            torch.device based on config.device setting
        """
        if self.config.device == "auto":
            return auto_device()
        return torch.device(self.config.device)

    def _ensure_training_device(self) -> None:
        """Move model to training device (GPU/MPS if available).

        Called at start of fit() to ensure training uses accelerator.
        """
        if self._model is None:
            return

        if self._training_device is None:
            self._training_device = self._resolve_device()

        current_device = next(self._model.parameters()).device
        if current_device != self._training_device:
            self._model.to(self._training_device)
            # Invalidate inference components (they cache the model's device)
            self._generator = None
            self._predictor = None
            self._embedder = None

    def _ensure_inference_device(self) -> None:
        """Move model to CPU for inference.

        CPU is faster for autoregressive generation due to:
        - No GPU kernel launch overhead per token
        - No CPU<->GPU memory transfer per step
        - Better single-threaded performance for sequential ops

        Once moved to CPU, the model stays there until fit() is called again.
        """
        if self._model is None:
            return

        current_device = next(self._model.parameters()).device
        if current_device.type != "cpu":
            self._model.to("cpu")
            # Invalidate inference components (they cache the model's device)
            self._generator = None
            self._predictor = None
            self._embedder = None

    def preprocess(
        self,
        data: list[dict],
        eval_data: list[dict] | None = None,
        verbose: bool = False,
    ) -> OrigamiPipeline:
        """Preprocess data and initialize model.

        For fresh training, this method:
        1. Sets up preprocessing based on numeric_mode
        2. Preprocesses data (fit_transform)
        3. Fits tokenizer to build vocabulary
        4. Creates model with correct configuration
        5. Stores preprocessed data for training

        When resuming from a checkpoint (model already loaded), this method:
        1. Transforms data using the already-fitted preprocessor
        2. Stores preprocessed data for training
        (Tokenizer and model are preserved from checkpoint)

        After calling preprocess(), call train() to train the model.

        Args:
            data: Training data as list of JSON-like dictionaries
            eval_data: Optional evaluation data for validation during training
            verbose: Whether to print info (vocab size, model params, device)

        Returns:
            self (for method chaining)
        """
        if not data:
            raise ValueError("Training data cannot be empty")

        # Check if resuming from checkpoint (model already loaded)
        if self._model is not None:
            # Resuming: just transform data using existing preprocessor
            if verbose:
                print("Resuming from checkpoint - using existing model and preprocessor")

            self._train_processed = self._transform_data(data)
            self._eval_processed = self._transform_data(eval_data) if eval_data else None
            return self

        # Fresh start: full preprocessing pipeline
        # Step 1: Setup and apply preprocessing
        train_processed, eval_processed = self._preprocess_data(data, eval_data)

        if verbose and self._preprocessor is not None:
            if isinstance(self._preprocessor, NumericScaler) and self._preprocessor.scaled_fields:
                print(f"Scaled fields ({len(self._preprocessor.scaled_fields)}):")
                for path in sorted(self._preprocessor.scaled_fields):
                    scaler = self._preprocessor.scalers[path]
                    mean = scaler.mean_[0]
                    std = scaler.scale_[0]
                    print(f"  - {path}: mean={mean:.4g}, std={std:.4g}")
            elif (
                isinstance(self._preprocessor, NumericDiscretizer)
                and self._preprocessor.discretized_fields
            ):
                print(f"Discretized fields ({len(self._preprocessor.discretized_fields)}):")
                for path in sorted(self._preprocessor.discretized_fields):
                    discretizer = self._preprocessor.discretizers[path]
                    n_bins = discretizer.n_bins_[0]
                    print(f"  - {path}: {n_bins} bins")

        # Step 2: Fit tokenizer on training data only.
        # Eval data uses the training vocabulary; unseen values/keys map to
        # UNK_VALUE/UNK_KEY. This prevents data leakage from eval into the
        # vocabulary and ensures schema masks handle eval UNK tokens properly.
        self._tokenizer = JSONTokenizer(
            max_depth=self.config.model.max_depth,
            max_array_index=self.config.model.max_array_position,
        )
        self._tokenizer.fit(train_processed, max_vocab_size=self.config.data.max_vocab_size)

        if verbose:
            print(f"Vocabulary size: {self._tokenizer.vocab.size}")
            if (
                self._tokenizer.pruning_stats
                and self._tokenizer.pruning_stats.num_values_pruned > 0
            ):
                stats = self._tokenizer.pruning_stats
                print(
                    f"  Pruned {stats.num_values_pruned} rare values "
                    f"(frequency threshold: {stats.value_frequency_threshold})"
                )

        # Step 3: Infer or set schema
        if self.config.data.infer_schema:
            from origami.constraints import SchemaDeriver

            deriver = SchemaDeriver()
            self._schema = deriver.derive(train_processed)
            if verbose:
                from origami.utils import format_schema

                print(f"Derived schema:\n{format_schema(self._schema)}")
        elif self.config.data.schema is not None:
            self._schema = self.config.data.schema
        else:
            self._schema = None

        # Step 4: Create model and move to training device
        self._model = self._create_model()
        self._training_device = self._resolve_device()
        self._model.to(self._training_device)

        if verbose:
            print(f"Model parameters: {self._model.get_num_parameters():,}")

        # Step 5: Store preprocessed data for train()
        self._train_processed = train_processed
        self._eval_processed = eval_processed

        # Reset inference components (model changed)
        self._generator = None
        self._predictor = None
        self._embedder = None

        return self

    def train(
        self,
        epochs: int | None = None,
        verbose: bool = False,
        callbacks: list | None = None,
    ) -> OrigamiPipeline:
        """Train the model on preprocessed data.

        Must call preprocess() first. Can be called multiple times to continue
        training the same model (model weights are preserved between calls).

        Args:
            epochs: Number of training epochs. Overrides config.training.num_epochs if provided.
            verbose: Whether to print training device info.
            callbacks: List of TrainerCallback instances for monitoring/customization.
                If None (default), uses [ProgressCallback()] for progress bars.
                Pass an explicit list to use only your callbacks (e.g., [] for silent).

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If preprocess() hasn't been called first.
        """
        from origami.training import OrigamiTrainer, ProgressCallback

        if self._train_processed is None:
            raise RuntimeError(
                "Must call preprocess() before train(). Alternatively, use fit() which calls both."
            )

        num_epochs = epochs if epochs is not None else self.config.training.num_epochs

        # Ensure model is on training device
        self._ensure_training_device()

        # Use training config, but override num_epochs if specified
        train_config = replace(self.config.training, num_epochs=num_epochs)

        # Build callbacks list: default to ProgressCallback if not specified
        if callbacks is None:
            all_callbacks = [ProgressCallback()]
        else:
            all_callbacks = list(callbacks)

        if verbose:
            print(f"Training device: {self._training_device}")

        # When device is "auto", pass None to trainer to allow accelerate integration
        # When device is explicitly specified, pass it to respect user's choice
        trainer_device = None if self.config.device == "auto" else self._training_device

        # Create inverse transform function for scaled numeric fields
        # This allows the trainer's evaluator to compute metrics on original scale
        inverse_fn = self._create_inverse_transform_fn()

        trainer = OrigamiTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_data=self._train_processed,
            eval_data=self._eval_processed,
            config=train_config,
            callbacks=all_callbacks if all_callbacks else None,
            device=trainer_device,
            training_state=self._training_state,  # Resume from checkpoint if available
            schema=self._schema,
            inverse_transform_fn=inverse_fn,
        )

        # Mark as fitted before training starts (all components are initialized)
        # This allows callbacks to save the pipeline during training
        self._fitted = True

        # Run training (handles KeyboardInterrupt gracefully)
        result = trainer.train()
        self._train_result = result

        # Capture training state for potential checkpoint save
        self._training_state = trainer.get_training_state()

        # Reset lazy-initialized inference components
        self._generator = None
        self._predictor = None
        self._embedder = None

        return self

    def fit(
        self,
        data: list[dict],
        eval_data: list[dict] | None = None,
        epochs: int | None = None,
        verbose: bool = False,
        callbacks: list | None = None,
    ) -> OrigamiPipeline:
        """Fit the pipeline on training data.

        Equivalent to calling preprocess() then train().

        Args:
            data: Training data as list of JSON-like dictionaries
            eval_data: Optional evaluation data for validation during training
            epochs: Number of training epochs. Overrides config.training.num_epochs if provided.
            verbose: Whether to print training info (vocab size, model params, device)
            callbacks: List of TrainerCallback instances for monitoring/customization.
                If None (default), uses [ProgressCallback()] for progress bars.
                Pass an explicit list to use only your callbacks (e.g., [] for silent).

        Returns:
            self (for method chaining)
        """
        self.preprocess(data, eval_data=eval_data, verbose=verbose)
        return self.train(epochs=epochs, verbose=verbose, callbacks=callbacks)

    def _preprocess_data(
        self,
        train_data: list[dict],
        eval_data: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict] | None]:
        """Apply preprocessing based on numeric_mode.

        Args:
            train_data: Training data
            eval_data: Optional evaluation data

        Returns:
            Tuple of (processed_train, processed_eval)
        """
        if self.config.data.numeric_mode == "disabled":
            # No preprocessing
            self._preprocessor = None
            return train_data, eval_data

        elif self.config.data.numeric_mode == "discretize":
            # Discretize high-cardinality numerics into bins
            self._preprocessor = NumericDiscretizer(
                cat_threshold=self.config.data.cat_threshold,
                n_bins=self.config.data.n_bins,
                strategy=self.config.data.bin_strategy,
            )
            train_processed = self._preprocessor.fit_transform(train_data)
            eval_processed = self._preprocessor.transform(eval_data) if eval_data else None
            return train_processed, eval_processed

        elif self.config.data.numeric_mode == "scale":
            # Scale high-cardinality numerics for continuous head
            self._preprocessor = NumericScaler(
                cat_threshold=self.config.data.cat_threshold,
            )
            train_processed = self._preprocessor.fit_transform(train_data)
            eval_processed = self._preprocessor.transform(eval_data) if eval_data else None
            return train_processed, eval_processed

        else:
            raise ValueError(f"Unknown numeric_mode: {self.config.data.numeric_mode}")

    def _create_model(self) -> OrigamiModel:
        """Create model with appropriate configuration."""
        assert self._tokenizer is not None, "Tokenizer must be fitted first"

        # Determine if continuous head is needed based on data config
        use_continuous_head = self.config.data.numeric_mode == "scale"

        # Create model config with continuous head setting based on data config
        model_config = replace(self.config.model, use_continuous_head=use_continuous_head)

        return OrigamiModel(model_config, self._tokenizer.vocab)

    def state_dict(self, include_training_state: bool = True) -> dict:
        """Get complete state dict for serialization.

        Returns a dictionary containing all state needed to reconstruct
        the pipeline: model weights, tokenizer, preprocessor, and config.
        Optionally includes training state for checkpoint resumption.

        Args:
            include_training_state: If True (default), include optimizer state,
                scheduler state, and training progress (epoch, global_step).
                Required for resuming training from a checkpoint.

        Returns:
            State dictionary suitable for torch.save()

        Raises:
            RuntimeError: If pipeline hasn't been fitted
        """
        self._check_fitted()

        state = {
            "version": "1.2",  # Bumped for schema support
            "config": asdict(self.config),
            "model_state_dict": self._model.state_dict(),
            "model_config": asdict(self._model.config),
            "tokenizer_state": self._tokenizer_to_dict(),
            "preprocessor_type": self._get_preprocessor_type(),
            "preprocessor_state": self._preprocessor_to_dict(),
            "schema": self._schema,
        }

        # Include training state if available and requested
        if include_training_state and self._training_state is not None:
            state["training_state"] = self._training_state

        return state

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> OrigamiPipeline:
        """Create pipeline from state dict.

        Reconstructs a complete pipeline from a state dictionary,
        including model, tokenizer, preprocessor, and config.
        If training state is present, it will be loaded for potential
        training resumption.

        Args:
            state_dict: State dictionary from state_dict() or torch.load()

        Returns:
            Loaded OrigamiPipeline ready for inference or training resumption
        """
        # Reconstruct config (nested dataclasses)
        config_dict = state_dict["config"]
        config = OrigamiConfig(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            device=config_dict["device"],
        )
        pipeline = cls(config)

        # Reconstruct preprocessor
        pipeline._preprocessor = cls._load_preprocessor(
            state_dict["preprocessor_type"],
            state_dict["preprocessor_state"],
        )

        # Reconstruct tokenizer
        pipeline._tokenizer = cls._tokenizer_from_dict(state_dict["tokenizer_state"])

        # Reconstruct model (stays on CPU - faster for inference)
        model_config = ModelConfig(**state_dict["model_config"])
        pipeline._model = OrigamiModel(model_config, pipeline._tokenizer.vocab)
        pipeline._model.load_state_dict(state_dict["model_state_dict"])
        pipeline._model.eval()

        # Recreate PDAs based on training config (trainer creates these during fit)
        if config.training.constrain_grammar:
            from origami.constraints.json_grammar import JSONGrammarPDA

            pipeline._model._grammar_pda = JSONGrammarPDA(
                pipeline._tokenizer.vocab, max_depth=model_config.max_depth
            )

        # Schema PDA is created after schema is loaded (below)

        # Set training device for potential future fit() calls
        pipeline._training_device = pipeline._resolve_device()

        # Load training state if present (for checkpoint resumption)
        if "training_state" in state_dict:
            pipeline._training_state = state_dict["training_state"]

        # Load schema if present
        pipeline._schema = state_dict.get("schema")

        # Create schema PDA if schema exists and constrain_schema was enabled
        if config.training.constrain_schema and pipeline._schema is not None:
            from origami.constraints import SchemaPDA

            pipeline._model._schema_pda = SchemaPDA(
                pipeline._schema,
                pipeline._tokenizer.vocab,
                max_depth=model_config.max_depth,
            )

        pipeline._fitted = True
        return pipeline

    def save(self, path: str | Path, include_training_state: bool = True) -> None:
        """Save the complete pipeline to a file.

        Saves model weights, tokenizer state, preprocessor state, and
        configuration in a single checkpoint file. Optionally includes
        training state for checkpoint resumption.

        Args:
            path: Path to save the checkpoint
            include_training_state: If True (default), include optimizer state,
                scheduler state, and training progress. Set to False for
                smaller inference-only checkpoints.

        Raises:
            RuntimeError: If pipeline hasn't been fitted
        """
        torch.save(self.state_dict(include_training_state=include_training_state), path)

    @classmethod
    def load(cls, path: str | Path) -> OrigamiPipeline:
        """Load a pipeline from a checkpoint file.

        Args:
            path: Path to the checkpoint

        Returns:
            Loaded OrigamiPipeline ready for inference
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_state_dict(state_dict)

    def predict(
        self,
        obj: dict,
        target_key: str,
        allow_complex_values: bool = False,
    ) -> Any:
        """Predict value for a target key.

        The object is preprocessed, prediction is made, and the result
        is inverse-transformed back to the original scale if applicable.

        Args:
            obj: JSON object. The target_key's current value is ignored.
            target_key: Key to predict (dot notation for nested keys)
            allow_complex_values: If False (default), restrict to primitive values
                only (strings, numbers, booleans, null). If True, allow objects
                and arrays which may require multiple tokens to generate.

        Returns:
            The predicted value
        """
        results = self.predict_batch([obj], target_key, allow_complex_values=allow_complex_values)
        return results[0]

    def predict_batch(
        self,
        objects: list[dict],
        target_key: str,
        batch_size: int = 32,
        allow_complex_values: bool = False,
        profile: bool = False,
    ) -> list[Any]:
        """Predict values for a batch of objects.

        Args:
            objects: List of JSON objects
            target_key: Key to predict (same for all objects)
            batch_size: Number of objects to process in parallel
            allow_complex_values: If False (default), restrict to primitive values
                only (strings, numbers, booleans, null). If True, allow objects
                and arrays which may require multiple tokens to generate.
            profile: If True, run with cProfile and print timing statistics.

        Returns:
            List of predicted values (one per object).
            Values are inverse-transformed to original scale if applicable.
        """
        self._check_fitted()

        # Preprocess objects
        processed = self._transform_data(objects)

        # Get or create predictor (has inverse_transform_fn configured if needed)
        predictor = self._get_predictor()

        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        # Run prediction (Predictor handles inverse transform internally)
        results = predictor.predict_batch(
            processed, target_key, batch_size=batch_size, allow_complex_values=allow_complex_values
        )

        if profile:
            profiler.disable()
            stream = StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumulative").print_stats(50)
            print(stream.getvalue())

        return results

    def predict_proba(
        self,
        obj: dict,
        target_key: str,
        values: list[Any] | None = None,
        top_k: int | None = None,
        allow_complex_values: bool = False,
    ) -> dict[Any, float] | list[tuple[Any, float]]:
        """Get probability distribution over possible values.

        Uses grammar-constrained probabilities from the model.

        Args:
            obj: JSON object
            target_key: Key to predict
            values: Specific values to get probabilities for
            top_k: If specified, return only top-k values sorted by probability
            allow_complex_values: If False (default), exclude OBJ_START/ARRAY_START
                from the probability distribution.

        Returns:
            If top_k is None: dict mapping values to probabilities
            If top_k is set: list of (value, prob) tuples, sorted desc by probability
        """
        self._check_fitted()

        # Preprocess object
        processed = self._transform_data([obj])[0]

        # Get or create predictor
        predictor = self._get_predictor()

        # Get probability distribution
        return predictor.predict_proba(
            processed,
            target_key,
            values=values,
            top_k=top_k,
            allow_complex_values=allow_complex_values,
        )

    def generate(
        self,
        num_samples: int = 1,
        batch_size: int = 32,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        allow_complex_values: bool = True,
    ) -> list[dict]:
        """Generate complete JSON objects.

        Returns objects with numeric values inverse-transformed to
        original scale if applicable.

        Args:
            num_samples: Number of objects to generate
            batch_size: Number of samples to generate in parallel
            max_length: Maximum sequence length
            temperature: Sampling temperature (1.0 = unchanged, <1.0 = more greedy)
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, sample from smallest set with cumulative prob >= top_p
            seed: Random seed for reproducibility
            allow_complex_values: If False, restrict field values to primitives only
                (no nested objects or arrays). Useful for untrained models. Default True.

        Returns:
            List of generated JSON objects
        """
        self._check_fitted()

        # Get or create generator
        generator = self._get_generator()

        # Generate samples
        samples = generator.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            allow_complex_values=allow_complex_values,
        )

        # Inverse transform numeric fields if needed
        if self._preprocessor is not None and self.config.data.numeric_mode == "scale":
            samples = [self._inverse_transform_object(s) for s in samples]

        return samples

    def evaluate(
        self,
        data: list[dict],
        target_key: str | None = None,
        metrics: dict[str, MetricSpec] | None = None,
        sample_size: int | None = None,
        batch_size: int = 32,
        allow_complex_values: bool | None = None,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Evaluate the model on data.

        Computes loss and any additional prediction-based metrics.

        Args:
            data: List of JSON objects to evaluate on
            target_key: Key to predict for prediction-based metrics.
                Falls back to config.target_key if not provided.
                Required if metrics are provided.
            metrics: Dict mapping prefixes to metric names or functions.
                Example: {"acc": "accuracy"}. Loss is always computed.
            sample_size: If set, randomly sample this many examples.
                None means use all data.
            batch_size: Batch size for evaluation.
            allow_complex_values: Whether to allow complex values (objects/arrays)
                during predictions. If None (default), auto-detected based on metrics.
            verbose: If True, show progress bars during evaluation.

        Returns:
            Dict mapping metric names to their values. Always includes "loss".

        Example:
            ```python
            # Just loss
            results = pipeline.evaluate(test_data)
            print(f"Loss: {results['loss']:.4f}")

            # Loss + accuracy
            results = pipeline.evaluate(
                test_data,
                target_key="label",
                metrics={"acc": "accuracy"},
            )
            print(f"Accuracy: {results['acc']:.2%}")
            ```
        """
        self._check_fitted()

        # Fall back to config target_key if not provided
        effective_target_key = target_key or self.config.training.target_key

        # Preprocess data
        processed = self._transform_data(data)

        # Move to CPU for faster evaluation
        self._ensure_inference_device()

        # Create inverse transform function if needed
        inverse_fn = None
        if isinstance(self._preprocessor, NumericScaler):
            inverse_fn = self._create_inverse_transform_fn()

        # Resolve allow_complex_values with auto-detection
        effective_allow_complex = self._resolve_allow_complex_values(allow_complex_values, metrics)

        # Create evaluator and run evaluation
        evaluator = OrigamiEvaluator(
            self._model,
            self._tokenizer,
            target_key=effective_target_key,
            inverse_transform=inverse_fn,
            allow_complex_values=effective_allow_complex,
            schema=self._schema,
            constrain_schema=self.config.training.constrain_schema,
        )

        return evaluator.evaluate(
            processed,
            metrics=metrics,
            sample_size=sample_size,
            batch_size=batch_size,
            verbose=verbose,
        )

    def embed(
        self,
        obj: dict,
        pooling: Literal["mean", "max", "last", "target"] = "mean",
        target_key: str | None = None,
        normalize: bool = True,
        enable_grad: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Get embedding for a JSON object.

        Args:
            obj: JSON object to embed
            pooling: Pooling strategy ("mean", "max", "last", "target")
            target_key: Required if pooling="target"
            normalize: Whether to L2-normalize the embedding
            enable_grad: If True, compute gradients and return tensor.
                If False (default), return numpy array for inference.

        Returns:
            Embedding as numpy array of shape (d_model,) if enable_grad=False,
            or torch.Tensor if enable_grad=True.
        """
        embeddings = self.embed_batch(
            [obj],
            pooling=pooling,
            target_key=target_key,
            normalize=normalize,
            enable_grad=enable_grad,
        )
        return embeddings[0]

    def embed_batch(
        self,
        objects: list[dict],
        pooling: Literal["mean", "max", "last", "target"] = "mean",
        target_key: str | None = None,
        normalize: bool = True,
        enable_grad: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Get embeddings for multiple JSON objects.

        Args:
            objects: List of JSON objects to embed
            pooling: Pooling strategy
            target_key: Required if pooling="target"
            normalize: Whether to L2-normalize embeddings
            enable_grad: If True, compute gradients and return tensor.
                If False (default), return numpy array for inference.

        Returns:
            Embeddings as numpy array of shape (batch_size, d_model) if enable_grad=False,
            or torch.Tensor if enable_grad=True.
        """
        self._check_fitted()

        # Preprocess objects
        processed = self._transform_data(objects)

        # Get or create embedder with appropriate pooling
        embedder = self._get_embedder(pooling)

        # Get embeddings
        embeddings = embedder.embed_batch(
            processed, target_key=target_key, normalize=normalize, enable_grad=enable_grad
        )

        # Return tensor for gradient computation, numpy for inference
        if enable_grad:
            return embeddings
        return embeddings.cpu().numpy()

    def _check_fitted(self) -> None:
        """Raise error if pipeline hasn't been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Pipeline must be fitted before use. Call fit() or load a checkpoint with load()."
            )

    def _transform_data(self, objects: list[dict]) -> list[dict]:
        """Transform data using the fitted preprocessor.

        Used for both training (when resuming from checkpoint) and inference.
        The preprocessor must already be fitted (from fit() or load()).

        Args:
            objects: Raw input objects

        Returns:
            Transformed objects ready for model input
        """
        if self._preprocessor is None:
            return objects

        if isinstance(self._preprocessor, NumericScaler):
            return self._preprocessor.transform(objects)
        elif isinstance(self._preprocessor, NumericDiscretizer):
            return self._preprocessor.transform(objects)
        else:
            return objects

    def _inverse_transform_object(self, obj: dict) -> dict:
        """Inverse transform all scaled numeric values in an object."""
        if not isinstance(self._preprocessor, NumericScaler):
            return obj

        return self._inverse_transform_value(obj, "")

    def _inverse_transform_value(self, value: Any, path: str) -> Any:
        """Recursively inverse transform values."""
        from origami.preprocessing.numeric_scaler import ScaledNumeric

        assert isinstance(self._preprocessor, NumericScaler)

        if isinstance(value, dict):
            return {
                key: self._inverse_transform_value(val, f"{path}.{key}" if path else key)
                for key, val in value.items()
            }
        elif isinstance(value, list):
            return [
                self._inverse_transform_value(item, f"{path}.{i}" if path else str(i))
                for i, item in enumerate(value)
            ]
        elif isinstance(value, ScaledNumeric):
            # This shouldn't happen in generated output, but handle it
            if path in self._preprocessor.scaled_fields:
                return self._preprocessor.inverse_transform_value(path, value.value)
            return value.value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Check if this path was a scaled field
            if path in self._preprocessor.scaled_fields:
                return self._preprocessor.inverse_transform_value(path, value)
            return value
        else:
            return value

    def _get_generator(self) -> OrigamiGenerator:
        """Get or create the generator.

        Moves model to CPU for faster inference if not already there.
        """
        self._ensure_inference_device()
        if self._generator is None:
            self._generator = OrigamiGenerator(self._model, self._tokenizer, schema=self._schema)
        return self._generator

    def _get_predictor(self) -> OrigamiPredictor:
        """Get or create the predictor with inverse transform configured.

        Moves model to CPU for faster inference if not already there.
        """
        self._ensure_inference_device()
        if self._predictor is None:
            inverse_fn = None
            if isinstance(self._preprocessor, NumericScaler):
                # Create inverse transform function for the predictor
                inverse_fn = self._create_inverse_transform_fn()

            self._predictor = OrigamiPredictor(
                self._model,
                self._tokenizer,
                inverse_transform_fn=inverse_fn,
                schema=self._schema,
            )
        return self._predictor

    def _create_inverse_transform_fn(self):
        """Create an inverse transform function for scaled numeric predictions.

        Returns:
            Function that takes (value, target_key) and returns the inverse-transformed value.
        """
        if not isinstance(self._preprocessor, NumericScaler):
            return None

        def inverse_transform(value, target_key: str):
            # Check if this field was scaled (using full path, not leaf key)
            if target_key not in self._preprocessor.scaled_fields:
                return value

            # Only transform numeric values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return self._preprocessor.inverse_transform_value(target_key, value)
            return value

        return inverse_transform

    def _resolve_allow_complex_values(
        self,
        explicit_value: bool | None,
        metrics: dict[str, MetricSpec] | None,
    ) -> bool:
        """Resolve allow_complex_values with auto-detection.

        Args:
            explicit_value: Explicit value passed to evaluate(), or None
            metrics: Metrics dict to check for complex-requiring metrics

        Returns:
            Resolved boolean value
        """
        from origami.training.metrics import any_metric_requires_complex_values

        if explicit_value is not None:
            return explicit_value

        return any_metric_requires_complex_values(metrics)

    def _get_embedder(self, pooling: Literal["mean", "max", "last", "target"]) -> OrigamiEmbedder:
        """Get or create an embedder with the specified pooling.

        Moves model to CPU for faster inference if not already there.
        """
        self._ensure_inference_device()
        # Always create a new embedder if pooling strategy differs
        if self._embedder is None or self._embedder.pooling != pooling:
            self._embedder = OrigamiEmbedder(self._model, self._tokenizer, pooling=pooling)
        return self._embedder

    # Serialization helpers

    def _tokenizer_to_dict(self) -> dict:
        """Serialize tokenizer state."""
        return {
            "vocab": self._tokenizer.vocab.to_dict(),
            "max_depth": self._tokenizer.max_depth,
            "max_array_index": self._tokenizer.max_array_index,
        }

    @staticmethod
    def _tokenizer_from_dict(data: dict) -> JSONTokenizer:
        """Reconstruct tokenizer from serialized state."""
        from origami.tokenizer.vocabulary import Vocabulary

        vocab = Vocabulary.from_dict(data["vocab"])
        return JSONTokenizer(
            vocab=vocab,
            max_depth=data["max_depth"],
            max_array_index=data["max_array_index"],
        )

    def _get_preprocessor_type(self) -> str | None:
        """Get preprocessor type name for serialization."""
        if self._preprocessor is None:
            return None
        return type(self._preprocessor).__name__

    def _preprocessor_to_dict(self) -> dict | None:
        """Serialize preprocessor state."""
        if self._preprocessor is None:
            return None

        if isinstance(self._preprocessor, NumericScaler):
            return self._preprocessor.to_dict()
        elif isinstance(self._preprocessor, NumericDiscretizer):
            return self._preprocessor.to_dict()
        else:
            raise ValueError(f"Unknown preprocessor type: {type(self._preprocessor)}")

    @staticmethod
    def _load_preprocessor(
        preprocessor_type: str | None,
        state: dict | None,
    ) -> NumericScaler | NumericDiscretizer | None:
        """Reconstruct preprocessor from serialized state."""
        if preprocessor_type is None or state is None:
            return None

        if preprocessor_type == "NumericScaler":
            return NumericScaler.from_dict(state)
        elif preprocessor_type == "NumericDiscretizer":
            return NumericDiscretizer.from_dict(state)
        else:
            raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"OrigamiPipeline(numeric_mode={self.config.data.numeric_mode!r}, {status})"
