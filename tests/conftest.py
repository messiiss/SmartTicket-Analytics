"""pytest 全局配置：把项目根目录加入 sys.path。

这样无需安装包（无 setup.py / pyproject）也能直接 ``pytest -q``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
