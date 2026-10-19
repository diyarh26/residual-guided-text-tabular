"""Frozen-only row caching and reference float32 PCA boundaries, without E5."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.features import FrozenE5Config
from refutab.pilot_features import CachedFrozenEncoder, FrozenRepresentation


class TextFixture:
    def __init__(self, state="base"):
        self.config = FrozenE5Config("explicit-test-fixture", max_length=16)
        self.state = state
        self.calls = []

    def assert_frozen(self):
        pass

    def state_sha256(self):
        return hashlib.sha256(self.state.encode()).hexdigest()

    def metadata(self):
        return {"config": asdict(self.config), "test_injection": True}

    def encode(self, frame, columns):
        self.calls.append(tuple(frame.index))
        result = np.empty((len(frame), len(columns), 384), dtype=np.float32)
        for field, column in enumerate(columns):
            for row, value in enumerate(frame[column]):
                seed = int.from_bytes(hashlib.sha256(f"{self.state}:{column}:{value}".encode()).digest()[:8], "big")
                vector = np.random.default_rng(seed).normal(size=384)
                result[row, field] = vector / np.linalg.norm(vector)
        return result


def text_frame(rows=70):
    return pd.DataFrame({"text": [f"private description {index}" for index in range(rows)],
                         "summary": [f"private summary {index % 13} - {index}" for index in range(rows)]},
                        index=[f"row-{index:03d}" for index in range(rows)])


class FrozenCacheTests(unittest.TestCase):
    def test_reuses_individual_rows_in_requested_order_and_never_caches_labels(self):
        frame = text_frame(8)
        frame["target"] = "private-label-marker"
        frame["numeric"] = range(8)
        with tempfile.TemporaryDirectory() as directory:
            source = TextFixture()
            cache = CachedFrozenEncoder(source, directory, allow_test_encoder=True)
            first = cache.encode(frame.iloc[[4, 0, 1, 2]], ("text", "summary"))
            second = cache.encode(frame.iloc[[2, 1, 3]], ("text", "summary"))
            self.assertEqual(source.calls, [("row-004", "row-000", "row-001", "row-002"), ("row-003",)])
            np.testing.assert_array_equal(second[:2], first[[3, 2]])
            other_source = TextFixture()
            reopened = CachedFrozenEncoder(other_source, directory, allow_test_encoder=True)
            changed_labels = frame.copy()
            changed_labels["target"] = "different-label"
            changed_labels["numeric"] = -999
            actual = reopened.encode(changed_labels.iloc[[3, 0, 4]], ("text", "summary"))
            self.assertEqual(other_source.calls, [])
            np.testing.assert_array_equal(actual, TextFixture().encode(frame.iloc[[3, 0, 4]], ("text", "summary")))
            self.assertEqual(reopened.metadata()["cache"]["cached_rows"], 5)
            self.assertEqual(len(list(Path(directory).rglob("embeddings.npz"))), 2)
            metadata_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(directory).rglob("*.json"))
            self.assertNotIn("private-label-marker", metadata_text)
            self.assertNotIn("private description", metadata_text)
            self.assertNotIn('"target"', metadata_text)
            self.assertIn('"row_id":["str","row-004"]', metadata_text)
            self.assertIn('"text_sha256"', metadata_text)

    def test_text_changes_and_typed_row_ids_create_distinct_safe_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            source = TextFixture()
            cache = CachedFrozenEncoder(source, directory, allow_test_encoder=True)
            frame = pd.DataFrame({"text": ["same text"]}, index=[1])
            a = cache.encode(frame, ("text",))
            changed = frame.copy()
            changed.loc[1, "text"] = "changed text"
            b = cache.encode(changed, ("text",))
            strings = frame.copy()
            strings.index = ["1"]
            c = cache.encode(strings, ("text",))
            self.assertFalse(np.array_equal(a, b))
            np.testing.assert_array_equal(a, c)
            self.assertEqual(source.calls, [(1,), (1,), ("1",)])
            self.assertEqual(cache.metadata()["cache"]["cached_rows"], 3)

    def test_exact_model_settings_and_environment_bindings_never_cross_reuse(self):
        frame = text_frame(3)
        with tempfile.TemporaryDirectory() as directory:
            source = TextFixture("base")
            original = CachedFrozenEncoder(source, directory, allow_test_encoder=True)
            original.encode(frame, ("text",))
            changed = TextFixture("different-weights")
            other = CachedFrozenEncoder(changed, directory, allow_test_encoder=True)
            other.encode(frame, ("text",))
            self.assertEqual(changed.calls, [tuple(frame.index)])
            self.assertNotEqual(original.binding_sha256, other.binding_sha256)
            configured = TextFixture()
            configured.config = replace(configured.config, max_length=24)
            configured_cache = CachedFrozenEncoder(configured, directory, allow_test_encoder=True)
            self.assertNotEqual(original.binding_sha256, configured_cache.binding_sha256)
            from refutab import pilot_features
            runtime = pilot_features._runtime_binding(TextFixture())
            runtime["torch"] = "deliberately-different-runtime"
            with patch("refutab.pilot_features._runtime_binding", return_value=runtime):
                environment_cache = CachedFrozenEncoder(TextFixture(), directory, allow_test_encoder=True)
                self.assertNotEqual(original.binding_sha256, environment_cache.binding_sha256)
            source.state = "mutated-in-place"
            with self.assertRaisesRegex(ValueError, "changed after cache binding"):
                original.encode(frame, ("text",))

    def test_shared_cache_requires_production_frozen_encoder_or_explicit_fixture_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "verified frozen E5"):
                CachedFrozenEncoder(TextFixture(), directory)
            self.assertFalse(list(Path(directory).iterdir()))
            with self.assertRaises(TypeError):
                CachedFrozenEncoder(TextFixture(), directory, allow_test_encoder="yes")

    def test_corrupt_pack_and_changed_metadata_fail_before_reuse(self):
        for artifact in ("embeddings.npz", "metadata.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                cache = CachedFrozenEncoder(TextFixture(), directory, allow_test_encoder=True)
                cache.encode(text_frame(3), ("text",))
                target = next(Path(directory).rglob(artifact))
                target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "checksum changed"):
                    cache.encode(text_frame(3), ("text",))
                if artifact == "embeddings.npz":
                    with self.assertRaisesRegex(ValueError, "embedding checksum mismatch"):
                        CachedFrozenEncoder(TextFixture(), directory, allow_test_encoder=True)

    def test_pending_packs_are_not_read_and_unexpected_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CachedFrozenEncoder(TextFixture(), directory, allow_test_encoder=True)
            pending = cache.namespace / (".pending-" + "a" * 32)
            pending.mkdir()
            (pending / "partial.bin").write_bytes(b"partial attempt")
            cache.encode(text_frame(2), ("text",))
            self.assertEqual(cache.metadata()["cache"]["pending_packs"], 1)
            self.assertEqual((pending / "partial.bin").read_bytes(), b"partial attempt")
            (cache.namespace / "unexpected.txt").write_text("review", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected cache entry"):
                cache.encode(text_frame(2), ("text",))

    def test_mutating_encoder_cannot_publish_pack(self):
        class MutatingFixture(TextFixture):
            def encode(self, frame, columns):
                value = super().encode(frame, columns)
                self.state = "mutated while encoding"
                return value

        with tempfile.TemporaryDirectory() as directory:
            cache = CachedFrozenEncoder(MutatingFixture(), directory, allow_test_encoder=True)
            with self.assertRaisesRegex(ValueError, "changed while encoding"):
                cache.encode(text_frame(2), ("text",))
            self.assertFalse(list(Path(directory).rglob("*.npz")))


class FrozenRepresentationTests(unittest.TestCase):
    def test_source_missing_policy_stringifies_before_fillna_for_fit_and_transform(self):
        class CapturingFixture(TextFixture):
            def __init__(self):
                super().__init__()
                self.frames = []

            def encode(self, frame, columns):
                self.frames.append(frame.copy())
                return super().encode(frame, columns)

        frame = text_frame(40)
        frame.iloc[0, 0], frame.iloc[1, 0], frame.iloc[2, 0] = np.nan, None, pd.NA
        source = CapturingFixture()
        fitted = FrozenRepresentation.fit(frame, ("text", "summary"), tuple(frame.index), source)
        expected = frame.apply(lambda column: column.astype(str).fillna(""))
        pd.testing.assert_frame_equal(source.frames[0], expected)
        self.assertEqual(source.frames[0]["text"].iloc[:3].tolist(), ["nan", "None", "<NA>"])
        fitted.transform(frame.iloc[:3])
        pd.testing.assert_frame_equal(source.frames[1], expected.iloc[:3])
        self.assertTrue(pd.isna(frame["text"].iloc[0]))
        self.assertIsNone(frame["text"].iloc[1])
        self.assertEqual(fitted.metadata()["text_value_conversion"], "pandas.Series.astype(str).fillna('')")

    def test_matches_original_float32_pca_math_and_fits_only_supplied_rows(self):
        frame = text_frame(70)
        source = TextFixture()
        fitted = FrozenRepresentation.fit(frame.iloc[:55], ("text", "summary"), tuple(frame.index[:55]), source)
        self.assertEqual(source.calls, [tuple(frame.index[:55])])
        self.assertEqual(fitted.fit_ids, tuple(frame.index[:55]))
        train = TextFixture().encode(frame.iloc[:55], ("text", "summary"))
        validation = TextFixture().encode(frame.iloc[55:], ("text", "summary"))
        references = [PCA(n_components=30, random_state=42).fit(np.ascontiguousarray(train[:, index])) for index in range(2)]
        for actual, expected in zip(fitted.models, references):
            self.assertEqual(actual.components_.dtype, np.float32)
            np.testing.assert_array_equal(actual.components_, expected.components_)
            np.testing.assert_array_equal(actual.mean_, expected.mean_)
        expected = np.concatenate([model.transform(np.ascontiguousarray(validation[:, index])) for index, model in enumerate(references)], axis=1)
        with patch("sklearn.decomposition.PCA.fit", side_effect=AssertionError("Transform fitted PCA")):
            actual = fitted.transform(frame.iloc[55:])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (15, 60))

    def test_native_roundtrip_preserves_float32_memory_order_and_predictions(self):
        frame = text_frame(70)
        fitted = FrozenRepresentation.fit(frame.iloc[:55], ("text", "summary"), tuple(frame.index[:55]), TextFixture())
        before = fitted.transform(frame.iloc[55:])
        with tempfile.TemporaryDirectory() as directory:
            path = fitted.save(Path(directory) / "representation")
            loaded = FrozenRepresentation.load(path, encoder=TextFixture())
            np.testing.assert_array_equal(before, loaded.transform(frame.iloc[55:]))
            self.assertEqual(loaded.fit_ids, fitted.fit_ids)
            self.assertEqual(loaded.metadata(), fitted.metadata())
            for original, restored in zip(fitted.models, loaded.models):
                self.assertEqual(original.components_.flags.f_contiguous, restored.components_.flags.f_contiguous)
            with np.load(path / "state.npz", allow_pickle=False) as arrays:
                self.assertTrue(all(arrays[name].dtype != object for name in arrays.files))
            with self.assertRaises(FileExistsError):
                fitted.save(path)

    def test_fit_ids_field_order_dimensions_and_dtype_are_enforced(self):
        frame = text_frame(35)
        with self.assertRaisesRegex(ValueError, "at least 30"):
            FrozenRepresentation.fit(frame.iloc[:29], ("text", "summary"), tuple(frame.index[:29]), TextFixture())
        with self.assertRaisesRegex(ValueError, "exactly match"):
            FrozenRepresentation.fit(frame, ("text", "summary"), tuple(reversed(frame.index)), TextFixture())
        with self.assertRaisesRegex(ValueError, "columns/order"):
            FrozenRepresentation.fit(frame[["summary", "text"]], ("text", "summary"), tuple(frame.index), TextFixture())
        fitted = FrozenRepresentation.fit(frame, ("text", "summary"), tuple(frame.index), TextFixture())
        with self.assertRaisesRegex(ValueError, "columns/order"):
            fitted.transform(frame[["summary", "text"]])
        with self.assertRaisesRegex(ValueError, "columns/order"):
            fitted.transform(frame.assign(target="forbidden"))

        class BadDtype(TextFixture):
            def encode(self, frame, columns):
                return super().encode(frame, columns).astype(np.float64)

        with self.assertRaisesRegex(ValueError, "finite float32"):
            FrozenRepresentation.fit(frame, ("text", "summary"), tuple(frame.index), BadDtype())

    def test_encoder_and_saved_fitting_provenance_mismatches_fail(self):
        frame = text_frame(35)
        fitted = FrozenRepresentation.fit(frame, ("text", "summary"), tuple(frame.index), TextFixture())
        with tempfile.TemporaryDirectory() as directory:
            path = fitted.save(Path(directory) / "representation")
            with self.assertRaisesRegex(ValueError, "encoder differs"):
                FrozenRepresentation.load(path, encoder=TextFixture("other-state"))
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["fit_ids"].reverse()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance checksum"):
                FrozenRepresentation.load(path, encoder=TextFixture())
        fitted.encoder.state = "changed-state"
        with self.assertRaisesRegex(ValueError, "encoder differs"):
            fitted.transform(frame)

    def test_constant_text_field_can_roundtrip_without_unused_variance_ratio_nan(self):
        frame = text_frame(35)
        frame["summary"] = "constant field"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted = FrozenRepresentation.fit(frame, ("text", "summary"), tuple(frame.index), TextFixture())
        before = fitted.transform(frame)
        self.assertTrue(np.isfinite(before).all())
        with tempfile.TemporaryDirectory() as directory:
            path = fitted.save(Path(directory) / "representation")
            restored = FrozenRepresentation.load(path, TextFixture())
            np.testing.assert_array_equal(before, restored.transform(frame))


if __name__ == "__main__":
    unittest.main()
