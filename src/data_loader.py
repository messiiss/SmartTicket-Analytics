"""数据加载模块。

职责：只负责“把原始文件读进来”，不做任何业务加工、不做静默清洗。
所有质量问题由 ``data_cleaner`` 统一记录，保证原始 JSON 永不被修改。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_TICKETS_PATH: Path = DEFAULT_DATA_DIR / "tickets.json"
DEFAULT_FIELDS_PATH: Path = DEFAULT_DATA_DIR / "ticket_fields.md"

#: 项目分析所依赖的字段清单（以 data/ticket_fields.md 为准）
EXPECTED_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "created_at",
    "category",
    "description",
    "priority",
    "resolution_time_hours",
    "satisfaction",
    "channel",
    "is_resolved",
)

#: 允许缺失的字段（缺失时降级处理，不阻断分析）
OPTIONAL_FIELDS: frozenset[str] = frozenset({"description", "channel"})


class TicketLoadError(RuntimeError):
    """工单数据无法加载时抛出，携带可读的排查提示。"""


@dataclass
class LoadResult:
    """加载结果：数据 + 加载阶段的元信息。"""

    dataframe: pd.DataFrame
    records: list[dict[str, Any]]
    source_path: Path
    missing_fields: list[str] = field(default_factory=list)
    unexpected_fields: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.dataframe)


def load_raw_records(path: str | Path | None = None) -> list[dict[str, Any]]:
    """读取 tickets.json 原始记录。

    Args:
        path: 工单 JSON 路径，默认 ``data/tickets.json``。

    Returns:
        原始字典列表，顺序与文件一致。

    Raises:
        TicketLoadError: 文件不存在、JSON 非法、或顶层不是列表时抛出。
    """
    target = Path(path) if path else DEFAULT_TICKETS_PATH
    if not target.exists():
        raise TicketLoadError(
            f"未找到工单数据文件：{target}。"
            "请确认附件 tickets.json 已放入 data/ 目录。"
        )

    try:
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise TicketLoadError(f"tickets.json 不是合法 JSON：{exc}") from exc

    if isinstance(payload, dict):
        # 兼容 {"tickets": [...]} 这类包装结构
        for key in ("tickets", "data", "items", "records"):
            if isinstance(payload.get(key), list):
                logger.info("检测到包装结构，使用 payload[%r] 作为工单列表", key)
                payload = payload[key]
                break

    if not isinstance(payload, list) or not payload:
        raise TicketLoadError("tickets.json 顶层应为非空列表，实际为空或类型不符。")

    if not all(isinstance(item, dict) for item in payload):
        raise TicketLoadError("tickets.json 中存在非字典元素，数据格式不符合预期。")

    logger.info("已加载 %d 条原始工单：%s", len(payload), target.name)
    return payload


def inspect_fields(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """对比实际字段与预期字段。

    Returns:
        (缺失的必需字段, 数据中出现的额外字段)
    """
    present = {key for record in records for key in record}
    missing = [f for f in EXPECTED_FIELDS if f not in present and f not in OPTIONAL_FIELDS]
    unexpected = sorted(present - set(EXPECTED_FIELDS))
    if missing:
        logger.warning("缺少必需字段：%s", missing)
    if unexpected:
        logger.info("发现未在字段说明中定义的字段：%s", unexpected)
    return missing, unexpected


def load_tickets(path: str | Path | None = None) -> LoadResult:
    """加载工单并转换为 DataFrame（仅做类型归位，不做业务清洗）。

    Args:
        path: 工单 JSON 路径，默认 ``data/tickets.json``。

    Returns:
        :class:`LoadResult`
    """
    records = load_raw_records(path)
    missing, unexpected = inspect_fields(records)

    dataframe = pd.DataFrame(records)
    # 补齐缺失字段，避免下游 KeyError；值保持为 NaN，由 cleaner 记录问题
    for field_name in EXPECTED_FIELDS:
        if field_name not in dataframe.columns:
            dataframe[field_name] = pd.NA

    ordered = list(EXPECTED_FIELDS) + [c for c in dataframe.columns if c not in EXPECTED_FIELDS]
    dataframe = dataframe[ordered]

    return LoadResult(
        dataframe=dataframe,
        records=records,
        source_path=Path(path) if path else DEFAULT_TICKETS_PATH,
        missing_fields=missing,
        unexpected_fields=unexpected,
    )


def load_field_documentation(path: str | Path | None = None) -> str:
    """读取字段说明 Markdown，供 Dashboard 展示。

    文件缺失时返回空字符串，不阻断主流程。
    """
    target = Path(path) if path else DEFAULT_FIELDS_PATH
    if not target.exists():
        logger.warning("未找到字段说明文件：%s", target)
        return ""
    return target.read_text(encoding="utf-8")
