"""时间趋势分析模块。

由于原始数据只覆盖 11 天（不足两个完整自然周），周趋势无法稳定成立，
因此本模块以「每日趋势」为主，并提供「时间前半段 vs 后半段」的日均对比作为补充。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MIN_DAYS_FOR_TREND: int = 3
MIN_DAYS_FOR_WEEKLY: int = 14


@dataclass
class TrendSummary:
    """趋势结论（全部为客观计算值）。"""

    date_start: str | None
    date_end: str | None
    days: int
    total_tickets: int
    daily_average: float | None
    peak_date: str | None
    peak_count: int | None
    first_half_daily_avg: float | None
    second_half_daily_avg: float | None
    change_ratio: float | None
    weekly_available: bool
    high_priority_daily_avg: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def daily_counts(df: pd.DataFrame, priority: str | None = None) -> pd.DataFrame:
    """按天统计工单量，补齐无工单的日期（值为 0）。

    Args:
        df: 清洗后的工单数据。
        priority: 只统计指定优先级，None 表示全部。

    Returns:
        含 ``date`` / ``count`` 的 DataFrame。
    """
    columns = ["date", "count"]
    if df is None or df.empty or "created_at" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    if priority is not None:
        working = working[working["priority"] == priority]
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["_date"] = pd.to_datetime(working["created_at"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["_date"])
    if working.empty:
        return pd.DataFrame(columns=columns)

    full_range = pd.date_range(working["_date"].min(), working["_date"].max(), freq="D")
    counts = working.groupby("_date").size().reindex(full_range, fill_value=0)
    return pd.DataFrame({"date": counts.index, "count": counts.to_numpy()})


def weekly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """按自然周统计工单量。数据不足两周时返回空表。"""
    columns = ["week", "count"]
    if df is None or df.empty or "created_at" not in df.columns:
        return pd.DataFrame(columns=columns)

    created = pd.to_datetime(df["created_at"], errors="coerce").dropna()
    if created.empty:
        return pd.DataFrame(columns=columns)
    span_days = (created.max().normalize() - created.min().normalize()).days + 1
    if span_days < MIN_DAYS_FOR_WEEKLY:
        logger.info("数据跨度仅 %d 天，跳过周趋势分析", span_days)
        return pd.DataFrame(columns=columns)

    weekly = created.dt.to_period("W").value_counts().sort_index()
    return pd.DataFrame({"week": weekly.index.astype(str), "count": weekly.to_numpy()})


def trend_summary(df: pd.DataFrame) -> TrendSummary:
    """汇总趋势指标，包含前后半段日均对比。"""
    daily = daily_counts(df)
    if daily.empty:
        return TrendSummary(None, None, 0, 0, None, None, None, None, None, None, False, None)

    days = len(daily)
    total = int(daily["count"].sum())
    peak_row = daily.loc[daily["count"].idxmax()]

    first_half = daily.iloc[: max(1, days // 2)]
    second_half = daily.iloc[max(1, days // 2):]
    first_avg = float(first_half["count"].mean())
    second_avg = float(second_half["count"].mean())
    change_ratio = (second_avg - first_avg) / first_avg if first_avg else None

    high_priority = daily_counts(df, priority="高")
    high_avg = float(high_priority["count"].mean()) if not high_priority.empty else None

    return TrendSummary(
        date_start=daily["date"].min().strftime("%Y-%m-%d"),
        date_end=daily["date"].max().strftime("%Y-%m-%d"),
        days=days,
        total_tickets=total,
        daily_average=round(total / days, 2) if days else None,
        peak_date=peak_row["date"].strftime("%Y-%m-%d"),
        peak_count=int(peak_row["count"]),
        first_half_daily_avg=round(first_avg, 2),
        second_half_daily_avg=round(second_avg, 2),
        change_ratio=None if change_ratio is None else round(change_ratio, 4),
        weekly_available=days >= MIN_DAYS_FOR_WEEKLY,
        high_priority_daily_avg=None if high_avg is None else round(high_avg, 2),
    )


def period_daily_rates(df: pd.DataFrame, split_date: str | pd.Timestamp) -> pd.DataFrame:
    """按切分日期对比不同维度的「日均工单量」。

    这是本项目识别「某类别突然增多」的核心方法：比较两个时间段内的日均产出，
    而不是直接比较绝对数量（两段天数不同，直接比数量会失真）。
    """
    columns = ["维度", "取值", "前半段数量", "后半段数量", "前半段日均", "后半段日均", "日均变化倍数"]
    if df is None or df.empty or "created_at" not in df.columns:
        return pd.DataFrame(columns=columns)

    split = pd.Timestamp(split_date).normalize()
    working = df.copy()
    working["_date"] = pd.to_datetime(working["created_at"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["_date"])
    if working.empty:
        return pd.DataFrame(columns=columns)

    start, end = working["_date"].min(), working["_date"].max()
    first_days = max(1, int((split - start).days))
    second_days = max(1, int((end - split).days) + 1)

    front = working[working["_date"] < split]
    back = working[working["_date"] >= split]

    rows: list[dict[str, Any]] = []
    for column, label in (("category", "问题类型"), ("priority", "优先级"), ("channel", "渠道")):
        if column not in working.columns:
            continue
        values = sorted(working[column].dropna().unique().tolist())
        for value in values:
            front_count = int((front[column] == value).sum())
            back_count = int((back[column] == value).sum())
            front_rate = front_count / first_days
            back_rate = back_count / second_days
            ratio = back_rate / front_rate if front_rate > 0 else None
            rows.append(
                {
                    "维度": label,
                    "取值": value,
                    "前半段数量": front_count,
                    "后半段数量": back_count,
                    "前半段日均": round(front_rate, 3),
                    "后半段日均": round(back_rate, 3),
                    "日均变化倍数": None if ratio is None else round(ratio, 2),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def suggest_split_date(df: pd.DataFrame) -> str | None:
    """取时间范围中点作为前后半段切分点。"""
    if df is None or df.empty or "created_at" not in df.columns:
        return None
    created = pd.to_datetime(df["created_at"], errors="coerce").dropna()
    if created.empty:
        return None
    middle = created.min().normalize() + (created.max().normalize() - created.min().normalize()) / 2
    return middle.strftime("%Y-%m-%d")
