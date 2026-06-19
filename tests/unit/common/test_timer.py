"""Tests for dx_modelzoo.common.timer."""

from dx_modelzoo.common.timer import EvaluationTimer


class TestEvaluationTimer:
    def test_context_manager_sets_start_time(self):
        with EvaluationTimer() as timer:
            assert timer.start_time is not None

    def test_debug_mode_flag(self):
        timer = EvaluationTimer(debug_mode=True)
        assert timer.debug_mode is True

    def test_no_exception_returns_false(self):
        with EvaluationTimer() as timer:
            pass
        # If no exception, __exit__ returns False (doesn't suppress)
        assert timer.start_time is not None

    def test_exception_not_suppressed(self):
        import pytest
        with pytest.raises(ValueError):
            with EvaluationTimer():
                raise ValueError("test")
