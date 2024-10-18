import unittest

from storm_ml.preprocessing.pipelines import build_estimation_pipeline, build_prediction_pipelines
from storm_ml.utils.common import SequenceOrderMethod


class TestBuildPipeline(unittest.TestCase):
    def test_build_prediction_pipelines(self):
        train_pipe, test_pipe = build_prediction_pipelines(
            target_field="foo", n_bins=10, sequence_order=SequenceOrderMethod.ORDERED, upscale=2
        )

        self.assertEqual(
            [list(p)[0] for p in list(train_pipe.get_params()["steps"])],
            ["binning", "schema", "tokenizer", "padding", "encoder"],
        )

        self.assertEqual(
            [list(p)[0] for p in list(test_pipe.get_params()["steps"])],
            ["binning", "target", "tokenizer", "padding", "encoder"],
        )

    def test_build_prediction_pipelines_with_permute(self):
        train_pipe, test_pipe = build_prediction_pipelines(
            target_field="foo", n_bins=10, sequence_order=SequenceOrderMethod.SHUFFLED, upscale=2
        )

        self.assertEqual(
            [list(p)[0] for p in list(train_pipe.get_params()["steps"])],
            ["binning", "schema", "upscaler", "permuter", "tokenizer", "padding", "encoder"],
        )

        self.assertEqual(
            [list(p)[0] for p in list(test_pipe.get_params()["steps"])],
            ["binning", "target", "tokenizer", "padding", "encoder"],
        )

    def test_build_estimation_pipeline_ordered(self):
        pipeline = build_estimation_pipeline(n_bins=10, sequence_order=SequenceOrderMethod.ORDERED, keep_id=False)

        self.assertEqual(
            [list(p)[0] for p in list(pipeline.get_params()["steps"])],
            ["binning", "schema", "exists", "tokenizer", "padding", "encoder"],
        )

    def test_build_estimation_pipeline_shuffled(self):
        pipeline = build_estimation_pipeline(n_bins=10, sequence_order=SequenceOrderMethod.SHUFFLED, keep_id=False)

        self.assertEqual(
            [list(p)[0] for p in list(pipeline.get_params()["steps"])],
            ["binning", "schema", "exists", "permuter", "tokenizer", "padding", "encoder"],
        )

    def test_build_estimation_pipeline_with_id(self):
        pipeline = build_estimation_pipeline(n_bins=10, sequence_order=SequenceOrderMethod.SHUFFLED, keep_id=True)

        self.assertEqual(
            [list(p)[0] for p in list(pipeline.get_params()["steps"])],
            ["binning", "id_setter", "schema", "exists", "permuter", "tokenizer", "padding", "encoder"],
        )
