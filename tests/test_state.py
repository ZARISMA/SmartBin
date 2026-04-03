"""Tests for smartwaste/state.py — AppState thread-safety and API contract."""

import threading
import time

import pytest

from smartwaste.state import AppState


class TestInitialValues:
    def test_initial_label_is_ready(self):
        state = AppState()
        label, _, _ = state.get_display()
        assert label == "Ready"

    def test_initial_detail_nonempty(self):
        state = AppState()
        _, detail, _ = state.get_display()
        assert isinstance(detail, str) and len(detail) > 0

    def test_initial_auto_classify_is_false(self):
        state = AppState()
        _, _, auto = state.get_display()
        assert auto is False

    def test_initial_last_capture_time_is_zero(self):
        state = AppState()
        assert state.last_capture_time == 0.0

    def test_initial_start_classify_succeeds(self):
        state = AppState()
        assert state.start_classify() is True


class TestGetDisplay:
    def test_returns_three_tuple(self, app_state):
        result = app_state.get_display()
        assert len(result) == 3

    def test_label_is_string(self, app_state):
        label, _, _ = app_state.get_display()
        assert isinstance(label, str)

    def test_detail_is_string(self, app_state):
        _, detail, _ = app_state.get_display()
        assert isinstance(detail, str)

    def test_auto_is_bool(self, app_state):
        _, _, auto = app_state.get_display()
        assert isinstance(auto, bool)


class TestSetStatus:
    def test_updates_label(self, app_state):
        app_state.set_status("Plastic", "A water bottle")
        label, _, _ = app_state.get_display()
        assert label == "Plastic"

    def test_updates_detail(self, app_state):
        app_state.set_status("Glass", "Green glass jar")
        _, detail, _ = app_state.get_display()
        assert detail == "Green glass jar"

    def test_does_not_change_auto(self, app_state):
        app_state.toggle_auto()
        app_state.set_status("Paper", "Cardboard")
        _, _, auto = app_state.get_display()
        assert auto is True

    def test_empty_strings_accepted(self, app_state):
        app_state.set_status("", "")
        label, detail, _ = app_state.get_display()
        assert label == ""
        assert detail == ""

    def test_unicode_values(self, app_state):
        app_state.set_status("Organic", "Ջերմուկ շիշ")
        _, detail, _ = app_state.get_display()
        assert detail == "Ջերմուկ շիշ"

    def test_overwrite_multiple_times(self, app_state):
        for label in ("Plastic", "Glass", "Paper", "Aluminum"):
            app_state.set_status(label, "detail")
        label, _, _ = app_state.get_display()
        assert label == "Aluminum"


class TestStartFinishClassify:
    def test_start_returns_true_when_idle(self, app_state):
        assert app_state.start_classify() is True

    def test_start_returns_false_when_busy(self, app_state):
        app_state.start_classify()
        assert app_state.start_classify() is False

    def test_start_returns_false_when_called_three_times(self, app_state):
        app_state.start_classify()
        assert app_state.start_classify() is False
        assert app_state.start_classify() is False

    def test_finish_allows_restart(self, app_state):
        app_state.start_classify()
        app_state.finish_classify()
        assert app_state.start_classify() is True

    def test_finish_without_start_does_not_raise(self, app_state):
        app_state.finish_classify()  # should not raise

    def test_double_finish_does_not_raise(self, app_state):
        app_state.start_classify()
        app_state.finish_classify()
        app_state.finish_classify()  # second finish also fine

    def test_multiple_start_finish_cycles(self, app_state):
        for _ in range(10):
            assert app_state.start_classify() is True
            app_state.finish_classify()


class TestToggleAuto:
    def test_toggle_off_to_on(self, app_state):
        result = app_state.toggle_auto()
        assert result is True

    def test_toggle_on_to_off(self, app_state):
        app_state.toggle_auto()
        result = app_state.toggle_auto()
        assert result is False

    def test_toggle_three_times(self, app_state):
        app_state.toggle_auto()
        app_state.toggle_auto()
        result = app_state.toggle_auto()
        assert result is True

    def test_toggle_reflected_in_get_display(self, app_state):
        app_state.toggle_auto()
        _, _, auto = app_state.get_display()
        assert auto is True

    def test_toggle_twice_back_to_false_in_get_display(self, app_state):
        app_state.toggle_auto()
        app_state.toggle_auto()
        _, _, auto = app_state.get_display()
        assert auto is False


class TestConcurrency:
    def test_only_one_start_classify_wins_under_contention(self):
        """Exactly one thread should win the start_classify race."""
        state = AppState()
        results = []
        lock = threading.Lock()

        def try_start():
            result = state.start_classify()
            with lock:
                results.append(result)

        threads = [threading.Thread(target=try_start) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 29

    def test_concurrent_set_status_no_exception(self):
        state = AppState()
        errors = []

        def write(i):
            try:
                state.set_status(f"Label{i}", f"Detail{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_reads_and_writes_no_exception(self):
        state = AppState()
        errors = []

        def writer():
            for i in range(200):
                try:
                    state.set_status(f"L{i}", f"D{i}")
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(200):
                try:
                    state.get_display()
                except Exception as e:
                    errors.append(e)

        threads = ([threading.Thread(target=writer)] +
                   [threading.Thread(target=reader) for _ in range(4)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_classify_released_allows_next_thread(self):
        """After finish_classify, a waiting thread should be able to claim it."""
        state = AppState()
        state.start_classify()

        second_result = []

        def try_after_release():
            time.sleep(0.02)
            second_result.append(state.start_classify())

        t = threading.Thread(target=try_after_release)
        t.start()
        state.finish_classify()
        t.join()

        assert second_result == [True]
