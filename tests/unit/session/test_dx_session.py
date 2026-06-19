"""Tests for dx_modelzoo.session.dx_session."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDxRuntimeSession:
    @patch.dict("sys.modules", {"dx_engine": MagicMock()})
    def test_import_error_without_dx_engine(self):
        # Reset module cache
        import sys
        if "dx_modelzoo.session.dx_session" in sys.modules:
            del sys.modules["dx_modelzoo.session.dx_session"]

        # With dx_engine mocked, should not raise ImportError
        from dx_modelzoo.session.dx_session import DxRuntimeSession
        assert DxRuntimeSession is not None
