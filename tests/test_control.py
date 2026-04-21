import pytest
from pydantic import ValidationError

from smartwaste.schemas import BinCommand, BinHeartbeat, WarningInfo
from smartwaste.state import AppState
from smartwaste.strategies import (
    STRATEGY_AUTO,
    STRATEGY_MANUAL,
    ManualStrategy,
    PresenceGateStrategy,
    build_strategy,
)


def test_build_strategy_manual():
    assert isinstance(build_strategy(STRATEGY_MANUAL), ManualStrategy)


def test_build_strategy_auto():
    assert isinstance(build_strategy(STRATEGY_AUTO), PresenceGateStrategy)


def test_build_strategy_unknown_defaults_manual():
    assert isinstance(build_strategy("nope"), ManualStrategy)
    assert isinstance(build_strategy(""), ManualStrategy)


def test_appstate_strategy_swap():
    s = AppState()
    assert s.take_pending_strategy_swap() is None
    s.request_strategy_swap("auto")
    assert s.take_pending_strategy_swap() == "auto"
    assert s.take_pending_strategy_swap() is None  # one-shot


def test_appstate_shutdown_and_restart():
    s = AppState()
    assert s.running is True
    assert s.shutdown_requested is False
    s.request_shutdown()
    assert s.shutdown_requested is True
    assert s.running is False
    assert s.restart_requested is False

    s2 = AppState()
    s2.request_restart()
    assert s2.restart_requested is True
    assert s2.shutdown_requested is True


def test_appstate_camera_count_and_warnings():
    s = AppState()
    s.set_camera_count(1)
    assert s.get_camera_count() == 1
    s.warnings.add("CAMERA_COUNT_LOW", "only 1 cam", severity="warning")
    items = s.warnings.list()
    assert items[0]["code"] == "CAMERA_COUNT_LOW"


def test_bin_command_valid_actions():
    for action in ("stop", "start", "restart", "set_strategy", "set_pipeline",
                   "classify", "toggle_auto", "clear_warnings"):
        BinCommand(action=action)  # no exception


def test_bin_command_invalid_action_rejected():
    with pytest.raises(ValidationError):
        BinCommand(action="explode")


def test_bin_heartbeat_accepts_new_fields():
    hb = BinHeartbeat(
        bin_id="bin-x",
        strategy="auto",
        pipeline="oak",
        camera_count=2,
        running=True,
        auto_classify=True,
        warnings=[WarningInfo(code="X", severity="info", message="hi")],
    )
    assert hb.strategy == "auto"
    assert hb.camera_count == 2
    assert hb.warnings[0].code == "X"


def test_bin_heartbeat_defaults():
    hb = BinHeartbeat(bin_id="bin-x")
    assert hb.running is True
    assert hb.warnings == []
    assert hb.camera_count == 0
