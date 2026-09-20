"""指标计算模块。

只做“客观指标计算”，不做任何主观打分（不设计“最重要问题评分”“最佳渠道”这类结论）。
所有函数对空数据、缺列、全空值均做保护。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from .data_cleaner import PRIORITY_ORDER

logger = logging.getLogger(__name__)


@dataclass
class OverallKPI:
    """首页核心指标。"""

    total_tickets: int
    resolved: int
    unresolved: int
    unresolved_rate: float
    avg_resolution_hours: float | None
    median_resolution_hours: float | None
    max_resolution_hours: float | None
    avg_satisfaction: float | None
    date_start: str | None
    date_end: str | None
    days: int
    high_priority_tickets: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_mean(series: pd.Series) -> float | None:
    """安全均值：空序列返回 None。"""
    if series is None or len(series) == 0:
        return None
    value = series.dropna()
    if value.empty:
        return None
    return float(value.mean())


def _safe_median(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.dropna()
    return None if value.empty else float(value.median())


def overall_kpis(df: pd.DataFrame) -> OverallKPI:
    """计算整体 KPI。"""
    if df is None or df.empty:
        logger.warning("空数据集：返回全零 KPI")
        return OverallKPI(0, 0, 0, 0.0, None, None, None, None, None, None, 0, 0)

    total = len(df)
    resolved_mask = df.get("is_resolved")
    resolved = int(resolved_mask.fillna(False).sum()) if resolved_mask is not None else 0
    unresolved = total - resolved

    created = pd.to_datetime(df["created_at"], errors="coerce") if "created_at" in df else None
    if created is not None and not created.dropna().empty:
        date_start = created.min().strftime("%Y-%m-%d")
        date_end = created.max().strftime("%Y-%m-%d")
        days = int((created.max().normalize() - created.min().normalize()).days) + 1
    else:
        date_start = date_end = None
        days = 0

    priority = df["priority"] if "priority" in df else pd.Series(dtype="object")
    high_priority = int((priority == "高").sum()) if not priority.empty else 0

    resolution = df["resolution_time_hours"] if "resolution_time_hours" in df else pd.Series(dtype=float)
    satisfaction = df["satisfaction"] if "satisfaction" in df else pd.Series(dtype=float)

    return OverallKPI(
        total_tickets=total,
        resolved=resolved,
        unresolved=unresolved,
        unresolved_rate=round(unresolved / total, 4) if total else 0.0,
        avg_resolution_hours=_round(_safe_mean(resolution)),
        median_resolution_hours=_round(_safe_median(resolution)),
        max_resolution_hours=_round(resolution.dropna().max()) if not resolution.dropna().empty else None,
        avg_satisfaction=_round(_safe_mean(satisfaction), 2),
        date_start=date_start,
        date_end=date_end,
        days=days,
        high_priority_tickets=high_priority,
    )


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def group_summary(df: pd.DataFrame, by: str, min_count: int = 1) -> pd.DataFrame:
    """按指定维度分组统计。

    Args:
        df: 清洗后的工单数据。
        by: 分组字段，如 ``category`` / ``priority`` / ``channel``。
        min_count: 低于该数量的分组会被保留，但调用方可据此提示“样本量小”。

    Returns:
        含 工单数 / 占比 / 平均处理时长 / 中位处理时长 / 平均满意度 / 未解决数 / 未解决率 的表格。
    """
    columns = [
        by,
        "工单数",
        "占比",
        "平均处理时长(h)",
        "中位处理时长(h)",
        "平均满意度",
        "未解决数",
        "未解决率",
    ]
    if df is None or df.empty or by not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working[by] = working[by].fillna("未知")
    total = len(working)

    grouped = working.groupby(by, dropna=False)
    summary = pd.DataFrame(
        {
            "工单数": grouped.size(),
            "平均处理时长(h)": grouped["resolution_time_hours"].mean().round(2),
            "中位处理时长(h)": grouped["resolution_time_hours"].median().round(2),
            "平均满意度": grouped["satisfaction"].mean().round(2),
            "未解决数": grouped["is_resolved"].apply(
                lambda s: int((~s.fillna(False).astype(bool)).sum())
            ),
        }
    ).reset_index()

    summary["占比"] = (summary["工单数"] / total).round(4)
    summary["未解决率"] = (summary["未解决数"] / summary["工单数"]).round(4)

    if by == "priority":
        summary["_order"] = summary[by].map(PRIORITY_ORDER).fillna(99)
        summary = summary.sort_values("_order").drop(columns="_order")
    else:
        summary = summary.sort_values("工单数", ascending=False)

    logger.debug("分组统计 by=%s -> %d 组", by, len(summary))
    return summary[columns].reset_index(drop=True)


def resolution_stats(df: pd.DataFrame) -> dict[str, Any]:
    """处理效率统计（均值 + 分位数，避免极端值误导）。"""
    series = df["resolution_time_hours"].dropna() if "resolution_time_hours" in df else pd.Series(dtype=float)
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "p90": None,
            "iqr_upper_fence": None,
            "long_tail_threshold": None,
        }
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    return {
        "count": int(series.size),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "p90": round(float(series.quantile(0.9)), 2),
        "iqr_upper_fence": round(q3 + 1.5 * iqr, 2),
        "long_tail_threshold": round(q3 + 1.5 * iqr, 2),
    }


def long_tail_tickets(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """筛出处理时长明显偏长的工单（IQR 上界法）。"""
    columns = ["ticket_id", "category", "priority", "resolution_time_hours", "satisfaction", "channel", "is_resolved"]
    if df is None or df.empty or "resolution_time_hours" not in df.columns:
        return pd.DataFrame(columns=columns)

    limit = threshold if threshold is not None else resolution_stats(df)["iqr_upper_fence"]
    if limit is None:
        return pd.DataFrame(columns=columns)

    subset = df[df["resolution_time_hours"] > limit].copy()
    subset = subset.sort_values("resolution_time_hours", ascending=False)
    keep = [c for c in columns if c in subset.columns]
    return subset[keep].reset_index(drop=True)


def satisfaction_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """满意度评分分布。"""
    columns = ["satisfaction", "工单数", "占比"]
    if df is None or df.empty or "satisfaction" not in df.columns:
        return pd.DataFrame(columns=columns)

    series = df["satisfaction"].dropna()
    if series.empty:
        return pd.DataFrame(columns=columns)

    counts = series.value_counts().sort_index()
    total = int(counts.sum())
    result = pd.DataFrame(
        {
            "satisfaction": counts.index.astype(int),
            "工单数": counts.to_numpy(),
        }
    )
    result["占比"] = (result["工单数"] / total).round(4)
    return result.reset_index(drop=True)


def low_satisfaction_tickets(df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    """筛选低满意度工单（默认 satisfaction <= 2）。"""
    columns = ["ticket_id", "category", "priority", "channel", "resolution_time_hours", "satisfaction", "is_resolved"]
    if df is None or df.empty or "satisfaction" not in df.columns:
        return pd.DataFrame(columns=columns)

    subset = df[df["satisfaction"] <= threshold].copy()
    subset = subset.sort_values(["satisfaction", "resolution_time_hours"], ascending=[True, False])
    keep = [c for c in columns if c in subset.columns]
    return subset[keep].reset_index(drop=True)


def unresolved_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """未解决工单列表（按优先级、创建时间排序）。"""
    columns = ["ticket_id", "created_at", "category", "priority", "channel", "resolution_time_hours", "satisfaction"]
    if df is None or df.empty or "is_resolved" not in df.columns:
        return pd.DataFrame(columns=columns)

    subset = df[~df["is_resolved"].fillna(False).astype(bool)].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    subset["_order"] = subset["priority"].map(PRIORITY_ORDER).fillna(99)
    subset = subset.sort_values(["_order", "created_at"])
    keep = [c for c in columns if c in subset.columns]
    return subset[keep].reset_index(drop=True)


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """维度与处理时长 / 满意度之间的相关性（Spearman，对样本量小更稳健）。

    说明：只描述相关性，不做因果推断。
    """
    columns = ["维度", "对比项", "样本数", "相关系数", "解读"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    priority_score = df["priority"].map(PRIORITY_ORDER) if "priority" in df.columns else None

    pairs: list[tuple[str, str, pd.Series, pd.Series]] = [
        ("优先级", "处理时长", priority_score, df.get("resolution_time_hours")),
        ("优先级", "满意度", priority_score, df.get("satisfaction")),
    ]
    for name, col in (("问题类型", "category"), ("渠道", "channel")):
        if col in df.columns:
            codes = df[col].astype("category").cat.codes
            pairs.append((name, "处理时长", codes, df.get("resolution_time_hours")))
            pairs.append((name, "满意度", codes, df.get("satisfaction")))

    pairs.append(("处理时长", "满意度", df.get("resolution_time_hours"), df.get("satisfaction")))

    for dimension, metric, left, right in pairs:
        if left is None or right is None:
            continue
        pair = pd.DataFrame({"x": left, "y": right}).dropna()
        if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            rows.append(
                {
                    "维度": dimension,
                    "对比项": metric,
                    "样本数": len(pair),
                    "相关系数": None,
                    "解读": "样本量或取值分布不足，未计算相关系数",
                }
            )
            continue
        coefficient = float(pair["x"].corr(pair["y"], method="spearman"))
        strength = _describe_strength(abs(coefficient))
        direction = "正相关" if coefficient > 0 else "负相关"
        rows.append(
            {
                "维度": dimension,
                "对比项": metric,
                "样本数": len(pair),
                "相关系数": round(coefficient, 3),
                "解读": f"{strength}{direction}（仅表示相关，不代表因果）",
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _describe_strength(value: float) -> str:
    if value < 0.2:
        return "几乎无"
    if value < 0.4:
        return "弱"
    if value < 0.6:
        return "中等"
    if value < 0.8:
        return "较强"
    return "强"


def satisfaction_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """维度 × 满意度交叉表（用于识别“某类问题满意度持续偏低”）。"""
    columns = [dimension, "工单数", "平均满意度", "低满意度数(≤2)", "低满意度率"]
    if df is None or df.empty or dimension not in df.columns or "satisfaction" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working[dimension] = working[dimension].fillna("未知")
    working["is_low"] = working["satisfaction"] <= 2

    grouped = working.groupby(dimension)
    result = pd.DataFrame(
        {
            "工单数": grouped.size(),
            "平均满意度": grouped["satisfaction"].mean().round(2),
            "低满意度数(≤2)": grouped["is_low"].sum().astype(int),
        }
    ).reset_index()
    result["低满意度率"] = (result["低满意度数(≤2)"] / result["工单数"]).round(4)
    return result[columns].sort_values("低满意度率", ascending=False).reset_index(drop=True)


def high_risk_categories(df: pd.DataFrame) -> pd.DataFrame:
    """识别“数量多 + 处理慢 + 满意度低”的问题类型。

    注意：这是对客观指标的交叉筛选，不是主观打分。
    """
    columns = ["category", "工单数", "平均处理时长(h)", "平均满意度", "未解决数", "命中特征"]
    summary = group_summary(df, "category")
    if summary.empty:
        return pd.DataFrame(columns=columns)

    volume_threshold = float(summary["工单数"].quantile(0.5))
    time_threshold = float(summary["平均处理时长(h)"].median())
    satisfaction_threshold = float(summary["平均满意度"].median())

    flags: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        hits: list[str] = []
        if pd.notna(row["工单数"]) and row["工单数"] >= volume_threshold:
            hits.append("数量偏多")
        if pd.notna(row["平均处理时长(h)"]) and row["平均处理时长(h)"] >= time_threshold:
            hits.append("处理偏慢")
        if pd.notna(row["平均满意度"]) and row["平均满意度"] <= satisfaction_threshold:
            hits.append("满意度偏低")
        if len(hits) >= 2:
            flags.append(
                {
                    "category": row["category"],
                    "工单数": int(row["工单数"]),
                    "平均处理时长(h)": row["平均处理时长(h)"],
                    "平均满意度": row["平均满意度"],
                    "未解决数": int(row["未解决数"]),
                    "命中特征": " + ".join(hits),
                }
            )

    result = pd.DataFrame(flags, columns=columns)
    if result.empty:
        return result
    return result.sort_values(["未解决数", "工单数"], ascending=False).reset_index(drop=True)
