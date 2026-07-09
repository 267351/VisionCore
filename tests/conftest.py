"""pytest fixtures。"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """返回项目根目录。"""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_dir():
    """临时目录，测试后自动清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
