"""Tests for hexabin/camera_config.py — transforms, validation, store, JSON."""

import cv2
import numpy as np
import pytest

from hexabin.camera_config import (
    CameraConfig,
    CameraConfigStore,
    apply_transform,
    default_config,
)
from hexabin.config import CROP_PERCENT


@pytest.fixture
def frame():
    # Distinct pixel values so flips/rotations are detectable.
    return np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)


# ─────────────────────────────────────────────────────────────────────────────
# default_config — must reproduce the legacy symmetric side-crop exactly
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultConfig:
    def test_crop_matches_crop_percent(self):
        cfg = default_config()
        assert cfg.rotation == 0
        assert cfg.flip_h is False and cfg.flip_v is False
        assert cfg.crop == (CROP_PERCENT, 0.0, 1.0 - CROP_PERCENT, 1.0)

    def test_apply_reproduces_legacy_symmetric_crop(self, frame):
        h, w = frame.shape[:2]
        left = int(w * CROP_PERCENT)
        right = int(w * (1 - CROP_PERCENT))
        legacy = frame[:, left:right]
        out = apply_transform(frame, default_config())
        assert out.shape == legacy.shape
        assert np.array_equal(out, legacy)


# ─────────────────────────────────────────────────────────────────────────────
# apply_transform — rotate / flip / crop
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyTransform:
    def test_none_config_returns_frame(self, frame):
        assert apply_transform(frame, None) is frame

    def test_identity(self, frame):
        cfg = CameraConfig(crop=(0, 0, 1, 1))
        assert np.array_equal(apply_transform(frame, cfg), frame)

    @pytest.mark.parametrize("rot", [90, 270])
    def test_rotation_swaps_dims(self, frame, rot):
        cfg = CameraConfig(rotation=rot, crop=(0, 0, 1, 1))
        out = apply_transform(frame, cfg)
        assert out.shape[:2] == (frame.shape[1], frame.shape[0])

    def test_rotation_180_keeps_dims(self, frame):
        out = apply_transform(frame, CameraConfig(rotation=180, crop=(0, 0, 1, 1)))
        assert out.shape == frame.shape
        assert np.array_equal(out, cv2.rotate(frame, cv2.ROTATE_180))

    def test_flip_h_matches_cv2(self, frame):
        out = apply_transform(frame, CameraConfig(flip_h=True, crop=(0, 0, 1, 1)))
        assert np.array_equal(out, cv2.flip(frame, 1))

    def test_flip_v_matches_cv2(self, frame):
        out = apply_transform(frame, CameraConfig(flip_v=True, crop=(0, 0, 1, 1)))
        assert np.array_equal(out, cv2.flip(frame, 0))

    def test_flip_both_matches_cv2(self, frame):
        out = apply_transform(frame, CameraConfig(flip_h=True, flip_v=True, crop=(0, 0, 1, 1)))
        assert np.array_equal(out, cv2.flip(frame, -1))

    def test_crop_slices_center(self, frame):
        cfg = CameraConfig(crop=(0.25, 0.5, 0.75, 1.0))
        out = apply_transform(frame, cfg)
        # 640*0.5 wide, 480*0.5 tall
        assert out.shape[0] == 240
        assert out.shape[1] == 320

    def test_rotate_then_crop_order(self, frame):
        # rotate 90 → dims 640x480, then crop left half → width 240
        cfg = CameraConfig(rotation=90, crop=(0.0, 0.0, 0.5, 1.0))
        out = apply_transform(frame, cfg)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        assert out.shape[0] == rotated.shape[0]
        assert out.shape[1] == rotated.shape[1] // 2


# ─────────────────────────────────────────────────────────────────────────────
# CameraConfig validation + (de)serialization
# ─────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_bad_rotation(self):
        with pytest.raises(ValueError):
            CameraConfig(rotation=45).validate()

    def test_rejects_out_of_range_crop(self):
        with pytest.raises(ValueError):
            CameraConfig(crop=(-0.1, 0, 1, 1)).validate()

    def test_rejects_inverted_crop(self):
        with pytest.raises(ValueError):
            CameraConfig(crop=(0.8, 0, 0.2, 1)).validate()

    def test_rejects_too_small_crop(self):
        with pytest.raises(ValueError):
            CameraConfig(crop=(0.0, 0.0, 0.02, 1.0)).validate()

    def test_accepts_valid(self):
        cfg = CameraConfig(rotation=270, flip_h=True, crop=(0.1, 0.1, 0.9, 0.9)).validate()
        assert cfg.rotation == 270

    def test_from_dict_roundtrip(self):
        src = CameraConfig(rotation=90, flip_v=True, crop=(0.2, 0.1, 0.8, 0.95))
        again = CameraConfig.from_dict(src.to_dict())
        assert again == src

    def test_from_dict_rejects_short_crop(self):
        with pytest.raises(ValueError):
            CameraConfig.from_dict({"crop": [0.1, 0.2]})

    def test_from_dict_defaults(self):
        cfg = CameraConfig.from_dict({})
        assert cfg.rotation == 0 and cfg.crop == (0.0, 0.0, 1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# CameraConfigStore
# ─────────────────────────────────────────────────────────────────────────────


class TestStore:
    def test_get_returns_default_when_unset(self):
        store = CameraConfigStore()
        assert store.get(0) == default_config()

    def test_set_get(self):
        store = CameraConfigStore()
        cfg = CameraConfig(rotation=180, crop=(0, 0, 1, 1))
        store.set(0, cfg)
        assert store.get(0) == cfg

    def test_set_validates(self):
        store = CameraConfigStore()
        with pytest.raises(ValueError):
            store.set(0, CameraConfig(rotation=33))

    def test_reset_reverts_to_default(self):
        store = CameraConfigStore()
        store.set(1, CameraConfig(rotation=90, crop=(0, 0, 1, 1)))
        store.reset(1)
        assert store.get(1) == default_config()

    def test_raw_frame_is_copied(self, frame):
        store = CameraConfigStore()
        store.set_raw(0, frame)
        got = store.get_raw(0)
        assert got is not frame
        assert np.array_equal(got, frame)

    def test_get_raw_none_when_unset(self):
        assert CameraConfigStore().get_raw(0) is None

    def test_dict_roundtrip(self):
        store = CameraConfigStore()
        store.set(0, CameraConfig(rotation=90, crop=(0, 0, 1, 1)))
        store.set(1, CameraConfig(flip_h=True, crop=(0.1, 0.1, 0.9, 0.9)))
        clone = CameraConfigStore()
        clone.load_dict(store.to_dict())
        assert clone.get(0) == store.get(0)
        assert clone.get(1) == store.get(1)

    def test_load_dict_skips_invalid(self):
        store = CameraConfigStore()
        store.load_dict({"cameras": {"0": {"rotation": 999}, "1": {"rotation": 90, "crop": [0, 0, 1, 1]}}})
        assert store.get(0) == default_config()  # invalid skipped → default
        assert store.get(1).rotation == 90

    def test_json_roundtrip(self, tmp_path):
        path = str(tmp_path / "cams.json")
        store = CameraConfigStore()
        store.set(0, CameraConfig(rotation=270, flip_v=True, crop=(0.15, 0.0, 0.85, 1.0)))
        store.save_json(path)
        clone = CameraConfigStore()
        clone.load_json(path)
        assert clone.get(0) == store.get(0)

    def test_load_missing_json_keeps_defaults(self, tmp_path):
        store = CameraConfigStore()
        store.load_json(str(tmp_path / "nope.json"))  # must not raise
        assert store.get(0) == default_config()
