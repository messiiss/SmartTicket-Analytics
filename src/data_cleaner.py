"""数据清洗与数据质量检查模块。

设计原则：
1. **不修改原始文件**：所有处理都在内存中的副本上进行。
2. **不静默修改**：任何类型转换、非法值处理都会写入 :class:`DataQualityIssue`。
3. **安全降级**：无法转换的值置为 NaN，让下游按“空值”处理，而不是整表报错。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IssueLevel = Literal["error", "warning", "info"]

#: 业务允许的取值白名单。数据中出现的其它取值会被记录为异常值。
ALLOWED_CATEGORIES: tuple[str, ...] = (
    "退款退货",
    "物流查询",
    "商品咨询",
    "账号问题",
    "支付问题",
    "投诉",
)
ALLOWED_PRIORITIES: tuple[str, ...] = ("高", "中", "低")
ALLOWED_CHANNELS: tuple[str, ...] = ("在线", "电话", "邮件")
SATISFACTION_MIN: int = 1
SATISFACTION_MAX: int = 5

#: 优先级排序权重，供各模块统一排序使用
PRIORITY_ORDER: dict[str, int] = {"高": 0, "中": 1, "低": 2}

_TRUE_TOKENS: frozenset[str] = frozenset({"true", "1", "是", "yes", "y"})
_FALSE_TOKENS: frozenset[str] = frozenset({"false", "0", "否", "no", "n"})


@dataclass
class DataQualityIssue:
    """单条数据质量问题记录。"""

    level: IssueLevel
    field_name: str
    message: str
    ticket_id: str | None = None
    affected_rows: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "field_name": self.field_name,
            "ticket_id": self.ticket_id,
            "affected_rows": self.affected_rows,
            "message": self.message,
        }


@dataclass
class CleanResult:
    """清洗结果。"""

    dataframe: pd.DataFrame
    issues: list[DataQualityIssue] = field(default_factory=list)
    raw_row_count: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def issues_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["level", "field_name", "ticket_id", "affected_rows", "message"]
            )
        return pd.DataFrame([issue.to_dict() for issue in self.issues])


def _parse_bool(value: Any) -> bool | None:
    """尽力把任意取值转换为布尔值，失败返回 None。"""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return None


def _parse_number(value: Any) -> float | None:
    """尽力把任意取值转换为数值，失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return None if (isinstance(value, float) and np.isnan(value)) else float(value)
    text = str(value).strip().replace("小时", "").replace("h", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_tickets(dataframe: pd.DataFrame) -> CleanResult:
    """清洗工单数据并记录全部数据质量问题。

    Args:
        dataframe: ``data_loader.load_tickets`` 返回的原始 DataFrame。

    Returns:
        :class:`CleanResult`，其中 ``dataframe`` 为清洗后的副本。
    """
    if dataframe is None or dataframe.empty:
        logger.warning("输入数据为空，清洗流程直接返回空结果")
        empty = pd.DataFrame() if dataframe is None else dataframe.copy()
        return CleanResult(dataframe=empty, issues=[], raw_row_count=0)

    df = dataframe.copy()
    issues: list[DataQualityIssue] = []
    raw_rows = len(df)

    # 1. ticket_id 重复检查
    if "ticket_id" in df.columns:
        duplicated = df["ticket_id"].duplicated(keep=False) & df["ticket_id"].notna()
        if duplicated.any():
            dup_ids = sorted(df.loc[duplicated, "ticket_id"].astype(str).unique())
            issues.append(
                DataQualityIssue(
                    level="error",
                    field_name="ticket_id",
                    message=f"存在重复工单编号：{', '.join(dup_ids)}。已保留首次出现的记录。",
                    affected_rows=int(duplicated.sum()),
                )
            )
            df = df.drop_duplicates(subset="ticket_id", keep="first").reset_index(drop=True)

    # 2. created_at 转 datetime
    if "created_at" in df.columns:
        original = df["created_at"].astype("string")
        parsed = pd.to_datetime(original, format="%Y-%m-%d %H:%M", errors="coerce")
        fallback = parsed.isna() & original.notna()
        if fallback.any():
            parsed_fallback = pd.to_datetime(
                original[fallback], errors="coerce", format="mixed"
            )
            parsed.loc[fallback] = parsed_fallback
        bad_time = parsed.isna()
        if bad_time.any():
            issues.append(
                DataQualityIssue(
                    level="error",
                    field_name="created_at",
                    message="以下工单创建时间无法解析，已置为空值并按缺失处理。",
                    ticket_id=", ".join(
                        df.loc[bad_time, "ticket_id"].astype(str).tolist()[:10]
                    ),
                    affected_rows=int(bad_time.sum()),
                )
            )
        df["created_at"] = parsed
        df["date"] = parsed.dt.date
    else:
        df["date"] = pd.NA

    # 3. 数值字段转换
    for column in ("resolution_time_hours", "satisfaction"):
        if column not in df.columns:
            continue
        converted = df[column].map(_parse_number)
        newly_na = converted.isna() & df[column].notna()
        if newly_na.any():
            issues.append(
                DataQualityIssue(
                    level="warning",
                    field_name=column,
                    message=f"{column} 中存在无法转换为数值的取值，已置为空值。",
                    affected_rows=int(newly_na.sum()),
                )
            )
        df[column] = pd.to_numeric(pd.Series(converted, index=df.index), errors="coerce")

    # 4. 缺失值统计
    for column in df.columns:
        missing = int(df[column].isna().sum())
        if missing:
            level: IssueLevel = "warning" if column in {"description", "channel"} else "error"
            issues.append(
                DataQualityIssue(
                    level=level,
                    field_name=column,
                    message=f"{column} 存在 {missing} 条缺失值。",
                    affected_rows=missing,
                )
            )

    # 5. satisfaction 合理区间
    if "satisfaction" in df.columns:
        invalid_sat = df["satisfaction"].notna() & (
            (df["satisfaction"] < SATISFACTION_MIN) | (df["satisfaction"] > SATISFACTION_MAX)
        )
        if invalid_sat.any():
            ids = df.loc[invalid_sat, "ticket_id"].astype(str).tolist()
            issues.append(
                DataQualityIssue(
                    level="warning",
                    field_name="satisfaction",
                    message=(
                        f"满意度评分超出 {SATISFACTION_MIN}-{SATISFACTION_MAX} 合理区间："
                        f"{', '.join(ids)}。已在满意度统计中排除该记录。"
                    ),
                    ticket_id=", ".join(ids),
                    affected_rows=int(invalid_sat.sum()),
                )
            )
            df.loc[invalid_sat, "satisfaction"] = np.nan

    # 6. resolution_time_hours 负数
    if "resolution_time_hours" in df.columns:
        negative = df["resolution_time_hours"].notna() & (df["resolution_time_hours"] < 0)
        if negative.any():
            ids = df.loc[negative, "ticket_id"].astype(str).tolist()
            issues.append(
                DataQualityIssue(
                    level="warning",
                    field_name="resolution_time_hours",
                    message=(
                        f"处理时长为负数：{', '.join(ids)}，不符合业务含义，"
                        "已按缺失值处理并排除出效率统计。"
                    ),
                    ticket_id=", ".join(ids),
                    affected_rows=int(negative.sum()),
                )
            )
            df.loc[negative, "resolution_time_hours"] = np.nan

    # 7. 分类字段异常值
    for column, allowed in (
        ("priority", ALLOWED_PRIORITIES),
        ("category", ALLOWED_CATEGORIES),
        ("channel", ALLOWED_CHANNELS),
    ):
        if column not in df.columns:
            continue
        text = df[column].astype("string").str.strip()
        df[column] = text
        unknown = text.notna() & ~text.isin(list(allowed))
        if unknown.any():
            values = sorted(text[unknown].dropna().unique().tolist())
            issues.append(
                DataQualityIssue(
                    level="warning",
                    field_name=column,
                    message=(
                        f"{column} 出现白名单外的取值：{values}。"
                        "该记录仍保留在总量统计中，但在分组统计中单独归类为「其它」。"
                    ),
                    affected_rows=int(unknown.sum()),
                )
            )

    # 8. is_resolved 类型检查
    if "is_resolved" in df.columns:
        converted = df["is_resolved"].map(_parse_bool)
        unconvertible = converted.isna() & df["is_resolved"].notna()
        if unconvertible.any():
            issues.append(
                DataQualityIssue(
                    level="warning",
                    field_name="is_resolved",
                    message="is_resolved 存在无法识别的取值，已置为空值。",
                    affected_rows=int(unconvertible.sum()),
                )
            )
        df["is_resolved"] = pd.Series(converted, index=df.index).astype("boolean")

    # 9. description 文本规范
    if "description" in df.columns:
        df["description"] = df["description"].astype("string").str.strip()
        empty_desc = df["description"].isna() | (df["description"] == "")
        if empty_desc.any():
            issues.append(
                DataQualityIssue(
                    level="warning",
                    field_name="description",
                    message="存在空的问题描述，相似问题分析会自动跳过这些工单。",
                    affected_rows=int(empty_desc.sum()),
                )
            )

    # 10. 数值字段出现小数（提示级，不影响分析）
    if "resolution_time_hours" in df.columns:
        non_integer = df["resolution_time_hours"].dropna() % 1 != 0
        if non_integer.any():
            issues.append(
                DataQualityIssue(
                    level="info",
                    field_name="resolution_time_hours",
                    message="处理时长包含小数（如 0.5 小时），统计时按原值参与计算。",
                    affected_rows=int(non_integer.sum()),
                )
            )

    logger.info(
        "清洗完成：输入 %d 条 -> 输出 %d 条，问题记录 %d 条（error=%d, warning=%d）",
        raw_rows,
        len(df),
        len(issues),
        sum(1 for i in issues if i.level == "error"),
        sum(1 for i in issues if i.level == "warning"),
    )
    return CleanResult(dataframe=df, issues=issues, raw_row_count=raw_rows)


def summarize_data_quality(clean_result: CleanResult) -> dict[str, Any]:
    """把数据质量问题汇总为可展示的结构化结果。"""
    df = clean_result.dataframe
    date_series = pd.to_datetime(df.get("created_at"), errors="coerce") if "created_at" in df else None
    return {
        "raw_rows": clean_result.raw_row_count,
        "clean_rows": len(df),
        "error_count": clean_result.error_count,
        "warning_count": clean_result.warning_count,
        "missing_cells": int(df.isna().sum().sum()) if not df.empty else 0,
        "duplicate_ticket_ids": int(
            df["ticket_id"].duplicated().sum() if "ticket_id" in df.columns else 0
        ),
        "date_start": None if date_series is None or date_series.dropna().empty else str(date_series.min()),
        "date_end": None if date_series is None or date_series.dropna().empty else str(date_series.max()),
        "issues": [issue.to_dict() for issue in clean_result.issues],
    }
