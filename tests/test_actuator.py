"""Tests for hexabin/actuator.py and the module-map config parsing."""

from unittest.mock import MagicMock, patch

import pytest

import hexabin.actuator as actuator
from hexabin.actuator import LogActuator, dispatch, get_actuator, resolve_module
from hexabin.config import DEFAULT_MODULE_MAP, _parse_module_map


@pytest.fixture(autouse=True)
def _reset_singleton():
    actuator._instance = None
    yield
    actuator._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# resolve_module / MODULE_MAP
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveModule:
    def test_default_map_covers_all_physical_categories(self):
        for category, module in DEFAULT_MODULE_MAP.items():
            assert resolve_module(category) == module

    def test_empty_has_no_module(self):
        assert resolve_module("Empty") is None

    def test_unknown_category_has_no_module(self):
        assert resolve_module("Unicorn") is None


class TestParseModuleMap:
    def test_blank_returns_default(self):
        assert _parse_module_map("") == DEFAULT_MODULE_MAP
        assert _parse_module_map("   ") == DEFAULT_MODULE_MAP

    def test_valid_override(self):
        assert _parse_module_map('{"Plastic": 3, "Glass": 1}') == {"Plastic": 3, "Glass": 1}

    def test_string_module_numbers_coerced(self):
        assert _parse_module_map('{"Plastic": "4"}') == {"Plastic": 4}

    def test_bad_json_returns_default(self):
        assert _parse_module_map("{not json") == DEFAULT_MODULE_MAP

    def test_non_object_returns_default(self):
        assert _parse_module_map("[1, 2]") == DEFAULT_MODULE_MAP

    def test_non_int_values_return_default(self):
        assert _parse_module_map('{"Plastic": "left"}') == DEFAULT_MODULE_MAP

    def test_empty_and_unknown_keys_dropped(self):
        parsed = _parse_module_map('{"Empty": 1, "Unicorn": 2, "Glass": 4}')
        assert parsed == {"Glass": 4}

    def test_only_invalid_keys_returns_default(self):
        assert _parse_module_map('{"Unicorn": 2}') == DEFAULT_MODULE_MAP

    def test_returns_copy_of_default(self):
        m = _parse_module_map("")
        m["Plastic"] = 99
        assert DEFAULT_MODULE_MAP["Plastic"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# get_actuator / dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestGetActuator:
    def test_default_is_log_actuator(self):
        assert isinstance(get_actuator(), LogActuator)

    def test_singleton(self):
        assert get_actuator() is get_actuator()

    def test_unknown_name_falls_back_to_log(self):
        with patch("hexabin.actuator.ACTUATOR", "warp-drive"):
            assert isinstance(get_actuator(), LogActuator)


class TestDispatch:
    def test_resolves_module_from_category(self):
        mock = MagicMock()
        with patch("hexabin.actuator.get_actuator", return_value=mock):
            dispatch("Plastic")
        mock.open_module.assert_called_once_with(1, "Plastic")

    def test_explicit_module_wins(self):
        mock = MagicMock()
        with patch("hexabin.actuator.get_actuator", return_value=mock):
            dispatch("Glass", module=5)
        mock.open_module.assert_called_once_with(5, "Glass")

    def test_empty_category_is_noop(self):
        mock = MagicMock()
        with patch("hexabin.actuator.get_actuator", return_value=mock):
            dispatch("Empty")
        assert not mock.open_module.called

    def test_actuator_exception_never_propagates(self):
        mock = MagicMock()
        mock.open_module.side_effect = RuntimeError("servo jam")
        with patch("hexabin.actuator.get_actuator", return_value=mock):
            dispatch("Plastic")  # must not raise

    def test_non_int_module_never_propagates(self):
        dispatch("Plastic", module="not-a-number")  # must not raise
