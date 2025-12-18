"""ORIGAMI Pipeline - end-to-end training and inference.

Provides a unified API for training ORIGAMI models on JSON data and
performing inference with automatic preprocessing and inverse transforms.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from origami.inference import OrigamiEmbedder, OrigamiEvaluator, OrigamiGenerator, OrigamiPredictor
from origami.inference.evaluator import MetricFn
from origami.model import OrigamiConfig, OrigamiModel
from origami.model.config import TrainingConfig
from origami.preprocessing import NumericDiscretizer, NumericScaler
from origami.tokenizer import JSONTokenizer
from origami.utils.device import auto_device

from .config import PipelineConfig

if TYPE_CHECKING:
    from origami.training import TrainResult


class OrigamiPipeline:
    """End-to-end ORIGAMI pipeline for training and inference.

    Combines preprocessing, tokenization, model training, and inference
    into a single unified API. Handles all the complexity of numeric
    scaling/discretization, vocabulary management, and checkpoint saving.

    Example - Training:
        ```python
        from origami import OrigamiPipeline

        # Minimal - just works with defaults
        pipeline = OrigamiPipeline()
        pipeline.fit(train_data, epochs=20)
        pipeline.save("model.pt")

        # With continuous head for high-cardinality numerics
        from origami import PipelineConfig
        config = PipelineConfig(numeric_mode="scale", d_model=128)
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

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration. Uses sensible defaults if None.
        """
        self.config = config or PipelineConfig()

        # Internal state - set during fit() or load()
        self._preprocessor: NumericScaler | NumericDiscretizer | None = None
        self._tokenizer: JSONTokenizer | None = None
        self._model: OrigamiModel | None = None
        self._fitted = False
        self._train_result: TrainResult | None = None

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

    def fit(
        self,
        data: list[dict],
        eval_data: list[dict] | None = None,
        epochs: int | None = None,
        verbose: bool = False,
        callbacks: list | None = None,
    ) -> OrigamiPipeline:
        """Fit the pipeline on training data.

        This method:
        1. Sets up preprocessing based on numeric_mode
        2. Preprocesses data
        3. Fits tokenizer to build vocabulary
        4. Creates model with correct configuration
        5. Trains the model

        Args:
            data: Training data as list of JSON-like dictionaries
            eval_data: Optional evaluation data for validation during training
            epochs: Number of training epochs. Overrides config if provided.
            verbose: Whether to print training info (vocab size, model params, device)
            callbacks: List of TrainerCallback instances for monitoring/customization.
                If None (default), uses [ProgressCallback()] for progress bars.
                Pass an explicit list to use only your callbacks (e.g., [] for silent).

        Returns:
            self (for method chaining)
        """
        from origami.training import OrigamiTrainer, ProgressCallback

        if not data:
            raise ValueError("Training data cannot be empty")

        num_epochs = epochs if epochs is not None else self.config.num_epochs

        # Step 1: Setup and apply preprocessing
        train_processed, eval_processed = self._preprocess_data(data, eval_data)

        # Step 2: Fit tokenizer on all preprocessed data
        all_processed = train_processed + (eval_processed or [])
        self._tokenizer = JSONTokenizer(
            max_depth=self.config.max_depth,
            max_array_index=self.config.max_array_position,
        )
        self._tokenizer.fit(all_processed)

        if verbose:
            print(f"Vocabulary size: {self._tokenizer.vocab.size}")

        # Step 3: Create model and move to training device
        self._model = self._create_model()
        self._training_device = self._resolve_device()
        self._model.to(self._training_device)

        if verbose:
            print(f"Model parameters: {self._model.get_num_parameters():,}")
            print(f"Training device: {self._training_device}")

        # Step 4: Create trainer and train
        train_config = TrainingConfig(
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=num_epochs,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            upscale_factor=self.config.upscale_factor,
            save_every_n_epochs=self.config.save_every_n_epochs,
            # Evaluation
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            eval_epochs=self.config.eval_epochs,
            eval_metrics=self.config.eval_metrics,
            eval_sample_size=self.config.eval_sample_size,
            eval_on_train=self.config.eval_on_train,
            target_key=self.config.target_key,
        )

        # Build callbacks list: default to ProgressCallback if not specified
        if callbacks is None:
            all_callbacks = [ProgressCallback()]
        else:
            all_callbacks = list(callbacks)

        trainer = OrigamiTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_data=train_processed,
            eval_data=eval_processed,
            config=train_config,
            shuffle=self.config.shuffle_keys,
            callbacks=all_callbacks if all_callbacks else None,
            device=self._training_device,
        )

        # Run training (handles KeyboardInterrupt gracefully)
        result = trainer.train()

        # Mark as fitted regardless of whether training completed or was interrupted
        self._fitted = True
        self._train_result = result

        # Reset lazy-initialized inference components
        self._generator = None
        self._predictor = None
        self._embedder = None

        return self

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
        if self.config.numeric_mode == "none":
            # No preprocessing
            self._preprocessor = None
            return train_data, eval_data

        elif self.config.numeric_mode == "discretize":
            # Discretize high-cardinality numerics into bins
            self._preprocessor = NumericDiscretizer(
                cat_threshold=self.config.cat_threshold,
                n_bins=self.config.n_bins,
                strategy=self.config.bin_strategy,
            )
            train_processed = self._preprocessor.fit_transform(train_data)
            eval_processed = self._preprocessor.transform(eval_data) if eval_data else None
            return train_processed, eval_processed

        elif self.config.numeric_mode == "scale":
            # Scale high-cardinality numerics for continuous head
            self._preprocessor = NumericScaler(
                cat_threshold=self.config.cat_threshold,
            )
            train_processed = self._preprocessor.fit_transform(train_data)
            eval_processed = self._preprocessor.transform(eval_data) if eval_data else None
            return train_processed, eval_processed

        else:
            raise ValueError(f"Unknown numeric_mode: {self.config.numeric_mode}")

    def _create_model(self) -> OrigamiModel:
        """Create model with appropriate configuration."""
        assert self._tokenizer is not None, "Tokenizer must be fitted first"

        # Determine if continuous head is needed
        use_continuous_head = self.config.numeric_mode == "scale"

        model_config = OrigamiConfig(
            vocab_size=self._tokenizer.vocab.size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            max_depth=self.config.max_depth,
            max_array_position=self.config.max_array_position,
            kvpe_pooling=self.config.kvpe_pooling,
            use_grammar_constraints=self.config.use_grammar_constraints,
            use_continuous_head=use_continuous_head,
            num_mixture_components=self.config.num_mixture_components,
        )

        return OrigamiModel(model_config, vocab=self._tokenizer.vocab)

    def save(self, path: str | Path) -> None:
        """Save the complete pipeline to a file.

        Saves model weights, tokenizer state, preprocessor state, and
        configuration in a single checkpoint file.

        Args:
            path: Path to save the checkpoint

        Raises:
            RuntimeError: If pipeline hasn't been fitted
        """
        self._check_fitted()

        checkpoint = {
            "version": "1.0",
            "config": asdict(self.config),
            "model_state_dict": self._model.state_dict(),
            "model_config": asdict(self._model.config),
            "tokenizer_state": self._tokenizer_to_dict(),
            "preprocessor_type": self._get_preprocessor_type(),
            "preprocessor_state": self._preprocessor_to_dict(),
        }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> OrigamiPipeline:
        """Load a pipeline from a checkpoint file.

        Args:
            path: Path to the checkpoint

        Returns:
            Loaded OrigamiPipeline ready for inference
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Reconstruct config
        config = PipelineConfig(**checkpoint["config"])
        pipeline = cls(config)

        # Reconstruct preprocessor
        pipeline._preprocessor = cls._load_preprocessor(
            checkpoint["preprocessor_type"],
            checkpoint["preprocessor_state"],
        )

        # Reconstruct tokenizer
        pipeline._tokenizer = cls._tokenizer_from_dict(checkpoint["tokenizer_state"])

        # Reconstruct model (stays on CPU - faster for inference)
        model_config = OrigamiConfig(**checkpoint["model_config"])
        pipeline._model = OrigamiModel(model_config, vocab=pipeline._tokenizer.vocab)
        pipeline._model.load_state_dict(checkpoint["model_state_dict"])
        pipeline._model.eval()

        # Set training device for potential future fit() calls
        pipeline._training_device = pipeline._resolve_device()

        pipeline._fitted = True
        return pipeline

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
    ) -> list[Any]:
        """Predict values for a batch of objects.

        Args:
            objects: List of JSON objects
            target_key: Key to predict (same for all objects)
            batch_size: Number of objects to process in parallel
            allow_complex_values: If False (default), restrict to primitive values
                only (strings, numbers, booleans, null). If True, allow objects
                and arrays which may require multiple tokens to generate.

        Returns:
            List of predicted values (one per object).
            Values are inverse-transformed to original scale if applicable.
        """
        self._check_fitted()

        # Preprocess objects
        processed = self._preprocess_for_inference(objects)

        # Get or create predictor (has inverse_transform_fn configured if needed)
        predictor = self._get_predictor()

        # Run prediction (Predictor handles inverse transform internally)
        return predictor.predict_batch(
            processed, target_key, batch_size=batch_size, allow_complex_values=allow_complex_values
        )

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
        processed = self._preprocess_for_inference([obj])[0]

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
        if self._preprocessor is not None and self.config.numeric_mode == "scale":
            samples = [self._inverse_transform_object(s) for s in samples]

        return samples

    def evaluate(
        self,
        data: list[dict],
        target_key: str | None = None,
        metrics: dict[str, MetricFn] | None = None,
        sample_size: int | None = None,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Evaluate the model on data.

        Computes loss and any additional prediction-based metrics.

        Args:
            data: List of JSON objects to evaluate on
            target_key: Key to predict for prediction-based metrics.
                Falls back to config.target_key if not provided.
                Required if metrics are provided.
            metrics: Dict mapping metric names to functions. Each function should
                follow sklearn convention: (y_true, y_pred) -> float.
                Example: {"acc": accuracy}. Loss is always computed.
            sample_size: If set, randomly sample this many examples.
                None means use all data.
            batch_size: Batch size for evaluation.

        Returns:
            Dict mapping metric names to their values. Always includes "loss".

        Example:
            ```python
            from origami.training import accuracy

            # Just loss
            results = pipeline.evaluate(test_data)
            print(f"Loss: {results['loss']:.4f}")

            # Loss + accuracy
            results = pipeline.evaluate(
                test_data,
                target_key="label",
                metrics={"acc": accuracy},
            )
            print(f"Accuracy: {results['acc']:.2%}")
            ```
        """
        self._check_fitted()

        # Fall back to config target_key if not provided
        effective_target_key = target_key or self.config.target_key

        # Preprocess data
        processed = self._preprocess_for_inference(data)

        # Move to CPU for faster evaluation
        self._ensure_inference_device()

        # Create inverse transform function if needed
        inverse_fn = None
        if isinstance(self._preprocessor, NumericScaler):
            inverse_fn = self._create_inverse_transform_fn()

        # Create evaluator and run evaluation
        evaluator = OrigamiEvaluator(
            self._model,
            self._tokenizer,
            target_key=effective_target_key,
            inverse_transform=inverse_fn,
        )

        return evaluator.evaluate(
            processed,
            metrics=metrics,
            sample_size=sample_size,
            batch_size=batch_size,
        )

    def embed(
        self,
        obj: dict,
        pooling: Literal["mean", "max", "last", "target"] = "mean",
        target_key: str | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Get embedding for a JSON object.

        Args:
            obj: JSON object to embed
            pooling: Pooling strategy ("mean", "max", "last", "target")
            target_key: Required if pooling="target"
            normalize: Whether to L2-normalize the embedding

        Returns:
            Embedding as numpy array of shape (d_model,)
        """
        embeddings = self.embed_batch(
            [obj], pooling=pooling, target_key=target_key, normalize=normalize
        )
        return embeddings[0]

    def embed_batch(
        self,
        objects: list[dict],
        pooling: Literal["mean", "max", "last", "target"] = "mean",
        target_key: str | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Get embeddings for multiple JSON objects.

        Args:
            objects: List of JSON objects to embed
            pooling: Pooling strategy
            target_key: Required if pooling="target"
            normalize: Whether to L2-normalize embeddings

        Returns:
            Embeddings as numpy array of shape (batch_size, d_model)
        """
        self._check_fitted()

        # Preprocess objects
        processed = self._preprocess_for_inference(objects)

        # Get or create embedder with appropriate pooling
        embedder = self._get_embedder(pooling)

        # Get embeddings
        embeddings = embedder.embed_batch(processed, target_key=target_key, normalize=normalize)

        return embeddings.cpu().numpy()

    def _check_fitted(self) -> None:
        """Raise error if pipeline hasn't been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Pipeline must be fitted before use. Call fit() or load a checkpoint with load()."
            )

    def _preprocess_for_inference(self, objects: list[dict]) -> list[dict]:
        """Apply preprocessing for inference.

        Args:
            objects: Raw input objects

        Returns:
            Preprocessed objects ready for model input
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
            self._generator = OrigamiGenerator(self._model, self._tokenizer)
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
            # Get the leaf key for inverse transform
            leaf_key = target_key.split(".")[-1]

            # Check if this field was scaled
            if leaf_key not in self._preprocessor.scaled_fields:
                return value

            # Only transform numeric values
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return self._preprocessor.inverse_transform_value(leaf_key, value)
            return value

        return inverse_transform

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
        return f"OrigamiPipeline(numeric_mode={self.config.numeric_mode!r}, {status})"
