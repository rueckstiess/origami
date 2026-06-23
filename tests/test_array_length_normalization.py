"""Tests for data-derived array-length normalization.

Covers the decoupling of array-length normalization (and enforcement) from
schema masking: the normalization divisor is a per-path maximum derived from
training data, shared identically by the collator (training) and generator
(inference), independent of constrain_schema.
"""

import torch

from origami import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OrigamiConfig,
    OrigamiPipeline,
    TrainingConfig,
)
from origami.inference.generator import OrigamiGenerator
from origami.model import OrigamiModel
from origami.model.heads import ContinuousHead
from origami.preprocessing import (
    array_length_cap_for_key,
    array_length_norm,
    derive_array_max_lengths,
)
from origami.preprocessing.array_length import array_length_norm_for_key
from origami.tokenizer import JSONTokenizer
from origami.tokenizer.path import IndexElement, KeyElement, normalize_path


class TestNormalizePath:
    def test_wildcards_for_indices(self):
        assert normalize_path(()) == ""
        assert normalize_path((KeyElement("name"),)) == "name"
        assert normalize_path((KeyElement("items"), IndexElement(0))) == "items.*"
        assert (
            normalize_path((KeyElement("items"), IndexElement(2), KeyElement("price")))
            == "items.*.price"
        )


class TestDeriveArrayMaxLengths:
    def test_single_path_max(self):
        data = [{"a": [1, 2, 3]}, {"a": [1]}, {"a": [1, 2, 3, 4, 5]}]
        assert derive_array_max_lengths(data) == {"a": 5}

    def test_nested_and_empty_arrays(self):
        data = [{"a": [], "b": {"c": [1, 2]}}, {"m": [[1], [2, 3, 4]]}]
        m = derive_array_max_lengths(data)
        assert m["a"] == 0  # empty array still recorded
        assert m["b.c"] == 2
        assert m["m"] == 2  # outer array has 2 elements
        assert m["m.*"] == 3  # inner arrays: max length 3

    def test_no_arrays(self):
        assert derive_array_max_lengths([{"x": 1, "y": "s"}]) == {}


class TestArrayLengthCap:
    def test_cap_is_observed_max(self):
        assert array_length_cap_for_key("a", {"a": 40}) == 40

    def test_cap_global_fallback(self):
        assert array_length_cap_for_key("zzz", {"a": 5, "b": 12}) == 12

    def test_cap_guards_against_zero(self):
        assert array_length_cap_for_key("a", {"a": 0}) == 1


class TestArrayLengthNorm:
    def test_norm_adds_headroom_above_cap(self):
        # 10% headroom: 40 -> +4 -> 44; norm is strictly above the cap.
        assert array_length_norm((KeyElement("a"),), {"a": 40}) == 44.0

    def test_norm_headroom_floor_of_one(self):
        # 10% of 5 rounds below 1 -> floored to +1 -> 6.
        assert array_length_norm((KeyElement("a"),), {"a": 5}) == 6.0

    def test_norm_always_exceeds_cap(self):
        for cap in (1, 2, 7, 13, 40, 100, 1000):
            m = {"a": cap}
            assert array_length_norm((KeyElement("a"),), m) > array_length_cap_for_key("a", m)

    def test_global_fallback_for_unknown_path(self):
        m = {"a": 5, "b": 12}
        # Unknown path falls back to the global max (12) plus headroom (+1).
        assert array_length_norm((KeyElement("zzz"),), m) == 13.0

    def test_empty_map_returns_one(self):
        assert array_length_norm((KeyElement("a"),), {}) == 1.0

    def test_key_helper_matches_path_helper(self):
        m = {"items": 7}
        path = (KeyElement("items"),)
        assert array_length_norm(path, m) == array_length_norm_for_key("items", m)


class TestContinuousInitDomain:
    def test_standard_init_keeps_means_near_zero(self):
        head = ContinuousHead(
            ModelConfig(use_continuous_head=True, num_mixture_components=10, continuous_init="standard")
        )
        mean_bias = head.proj.bias[10:20]
        # Default nn.Linear init: small, near zero, not spread across [0, 1].
        assert mean_bias.abs().max().item() < 0.5

    def test_unit_init_spreads_and_localizes_components(self):
        k = 10
        head = ContinuousHead(
            ModelConfig(use_continuous_head=True, num_mixture_components=k, continuous_init="unit")
        )
        mean_bias = head.proj.bias[k : 2 * k]
        log_var_bias = head.proj.bias[2 * k : 3 * k]
        assert torch.allclose(mean_bias, torch.linspace(0.0, 1.0, k))
        # Logistic scale exp(0.5 * log_var) ~ component spacing (localized).
        scale = torch.exp(0.5 * log_var_bias[0]).item()
        assert abs(scale - 1.0 / (k - 1)) < 1e-5

    def test_unit_init_single_component_no_op(self):
        # k == 1 must not crash (no spacing).
        ContinuousHead(
            ModelConfig(use_continuous_head=True, num_mixture_components=1, continuous_init="unit")
        )

    def test_pipeline_uses_unit_init_for_array_lengths(self):
        config = OrigamiConfig(
            data=DataConfig(model_array_lengths=True, numeric_mode="disabled"),
            model=ModelConfig(d_model=32, n_heads=4, n_layers=1),
        )
        p = OrigamiPipeline(config)
        p.preprocess([{"items": [1, 2, 3]}, {"items": [1]}])
        assert p._model.config.continuous_init == "unit"

    def test_pipeline_uses_standard_init_for_scale_mode(self):
        # Scale mode (standardized numerics) keeps the default init even with
        # array-length modeling on.
        config = OrigamiConfig(
            data=DataConfig(model_array_lengths=True, numeric_mode="scale", cat_threshold=1),
            model=ModelConfig(d_model=32, n_heads=4, n_layers=1),
        )
        p = OrigamiPipeline(config)
        p.preprocess([{"items": [1, 2, 3], "v": 1.0}, {"items": [1], "v": 9.0}])
        assert p._model.config.continuous_init == "standard"


class TestCollatorNormalizationDecoupled:
    def test_normalization_uses_map_not_schema(self):
        from origami.training.collator import OrigamiDataCollator

        data = [{"items": [1, 2, 3]}]
        tokenizer = JSONTokenizer()
        tokenizer.fit(data)
        collator = OrigamiDataCollator(
            tokenizer, model_array_lengths=True, array_max_lengths={"items": 6}
        )
        batch = collator([tokenizer.tokenize(obj) for obj in data])
        pos = (batch.input_ids == tokenizer.vocab.array_start_id).nonzero()[0].tolist()
        b, p = pos
        # Normalized by the buffered divisor (cap 6 + headroom 1 = 7): 3/7, step 1/7.
        norm = array_length_norm((KeyElement("items"),), {"items": 6})
        assert abs(batch.numeric_values[b, p].item() - 3 / norm) < 1e-6
        assert abs(batch.discretization_step[b, p].item() - 1 / norm) < 1e-6


class TestSampleIntegerNormVsCap:
    def test_cap_below_grid_is_respected(self):
        head = ContinuousHead(ModelConfig(use_continuous_head=True, num_mixture_components=1))
        weights = torch.tensor([[[1.0]]])
        means = torch.tensor([[[1.0]]])  # mass at top of [0,1]
        log_vars = torch.tensor([[[-6.0]]])  # sharp
        norm = torch.tensor([[10.0]])  # grid 0..10
        cap = torch.tensor([[3.0]])  # but capped at 3
        for seed in range(25):
            torch.manual_seed(seed)
            s = head.sample_integer(weights, means, log_vars, norm, max_values=cap)
            assert s.item() <= 3


class TestLengthEnforcementMask:
    def _generator(self):
        tokenizer = JSONTokenizer()
        tokenizer.fit([{"items": [1, 2, 3]}])
        model = OrigamiModel(
            ModelConfig(use_continuous_head=True, num_mixture_components=3), tokenizer.vocab
        )
        return OrigamiGenerator(model, tokenizer, constrain_grammar=False, constrain_schema=False)

    def test_suppresses_array_end_before_target(self):
        gen = self._generator()
        vocab = gen.tokenizer.vocab
        logits = torch.zeros(1, vocab.size)
        gen._apply_length_enforcement(logits, [["array"]], [[1]], [[3]])
        assert logits[0, vocab.array_end_id] == float("-inf")
        # value tokens remain available
        assert logits[0, next(iter(vocab._value_ids))] == 0.0

    def test_forces_array_end_at_target(self):
        gen = self._generator()
        vocab = gen.tokenizer.vocab
        logits = torch.zeros(1, vocab.size)
        gen._apply_length_enforcement(logits, [["array"]], [[3]], [[3]])
        assert logits[0, vocab.array_end_id] == 0.0
        for vid in vocab._value_ids:
            assert logits[0, vid] == float("-inf")
        assert logits[0, vocab.array_start_id] == float("-inf")

    def test_noop_when_not_in_array(self):
        gen = self._generator()
        vocab = gen.tokenizer.vocab
        logits = torch.zeros(1, vocab.size)
        gen._apply_length_enforcement(logits, [["object"]], [[]], [[]])
        assert (logits == 0.0).all()


class TestPipelineIntegration:
    def _config(self, constrain_schema):
        return OrigamiConfig(
            data=DataConfig(infer_schema=True, model_array_lengths=True),
            model=ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0),
            training=TrainingConfig(
                shuffle_keys=False,
                batch_size=32,
                warmup_steps=5,
                learning_rate=1e-3,
                constrain_grammar=True,
                constrain_schema=constrain_schema,
            ),
            inference=InferenceConfig(constrain_grammar=True, constrain_schema=True),
            device="cpu",
        )

    def test_map_independent_of_constrain_schema(self):
        # The derived map must be identical whether or not schema masking is on.
        data = [{"items": list(range(k))} for k in [3, 7, 2, 9, 5]]
        maps = []
        for cs in (True, False):
            p = OrigamiPipeline(self._config(cs))
            p.preprocess(data)
            maps.append(dict(p._model._array_max_lengths))
        assert maps[0] == maps[1] == {"items": 9}

    def test_save_load_round_trips_map(self, tmp_path):
        data = [{"items": list(range(k))} for k in [2, 4, 6, 3]]
        p = OrigamiPipeline(self._config(constrain_schema=False))
        p.fit(data, epochs=1, verbose=False, callbacks=[])
        assert p._model._array_max_lengths  # non-empty
        fn = tmp_path / "m.pt"
        p.save(str(fn))
        loaded = OrigamiPipeline.load(str(fn))
        assert loaded._model._array_max_lengths == p._model._array_max_lengths

    def test_load_pre_1_4_checkpoint_without_map(self):
        data = [{"items": list(range(k))} for k in [2, 4, 6, 3]]
        p = OrigamiPipeline(self._config(constrain_schema=False))
        p.fit(data, epochs=1, verbose=False, callbacks=[])
        state = p.state_dict()
        del state["array_max_lengths"]  # simulate old checkpoint
        loaded = OrigamiPipeline.from_state_dict(state)
        assert loaded._model._array_max_lengths == {}
