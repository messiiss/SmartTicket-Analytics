"""异常检测模块（项目核心）。

设计原则：
1. 每条异常必须带有 **evidence（判断依据）** 与 **severity（严重程度）**，
   没有依据的结论不输出。
2. 统计方法优先使用 **稳健统计**（IQR、中位数、前后半段日均对比），
   因为本数据集只有 50 条，均值 + 标准差容易被极端值带偏。
3. 措辞统一使用「异常信号 / 建议进一步确认」，
   不把统计偏离描述为确定的业务故障。

severity 的判定是**规则化**的（不是主观打分）：
- 高  ：直接影响用户资金/账号安全，或高优先级工单长期未闭环
- 关注：指标明显偏离（超过稳健阈值），但样本量有限，需人工确认
- 提示：轻度偏离或趋势性变化，用于持续观察
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from .data_cleaner import PRIORITY_ORDER
from .trend_analysis import daily_counts, period_daily_rates, suggest_split_date

logger = logging.getLogger(__name__)

Severity = Literal["高", "关注", "提示"]

# 阈值均为经验参数，可用 Dashboard 侧参数调整
DAILY_SPIKE_Z_THRESHOLD: float = 1.5
CATEGORY_GROWTH_RATIO: float = 2.0
CATEGORY_GROWTH_MIN_COUNT: int = 3
LOW_SATISFACTION_THRESHOLD: int = 2
HIGH_PRIORITY_UNRESOLVED_MIN: int = 1
LONG_RESOLUTION_MIN_HOURS: float = 24.0
REPEAT_CLUSTER_MIN_SIZE: int = 2


@dataclass
class AnomalySignal:
    """一条异常信号，结构与测评要求一致。"""

    type: str
    description: str
    evidence: str
    severity: Severity
    related_tickets: list[str] = field(default_factory=list)
    metric: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_daily_volume_anomaly(df: pd.DataFrame) -> list[AnomalySignal]:
    """异常 A：单日工单量明显高于近期平均水平。"""
    daily = daily_counts(df)
    if daily.empty or len(daily) < 3:
        return []

    counts = daily["count"].astype(float)
    mean, std = float(counts.mean()), float(counts.std(ddof=0))
    median = float(counts.median())
    q1, q3 = float(counts.quantile(0.25)), float(counts.quantile(0.75))
    upper_fence = q3 + 1.5 * (q3 - q1) if q3 > q1 else max(median * 2, median + 1)

    signals: list[AnomalySignal] = []
    for _, row in daily.iterrows():
        value = float(row["count"])
        z_score = (value - mean) / std if std > 0 else 0.0
        if z_score < DAILY_SPIKE_Z_THRESHOLD and value <= upper_fence:
            continue
        day = row["date"].strftime("%Y-%m-%d")
        ticket_ids = (
            df[pd.to_datetime(df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d") == day]["ticket_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        peak_category = (
            df[pd.to_datetime(df["created_at"], errors="coerce").dt.strftime("%Y-%m-%d") == day]["category"]
            .value_counts()
            .head(2)
            .to_dict()
        )
        signals.append(
            AnomalySignal(
                type="工单量异常",
                description=f"{day} 当日工单量 {int(value)} 条，高于该周期日均水平（{mean:.2f} 条/天）。",
                evidence=(
                    f"z-score={z_score:.2f}（阈值 {DAILY_SPIKE_Z_THRESHOLD}），"
                    f"当日数量 {int(value)}，周期中位数 {median:.0f}，IQR 上界 {upper_fence:.1f}；"
                    f"当日主要问题类型：{peak_category}。"
                ),
                severity="关注" if z_score >= 2 else "提示",
                related_tickets=ticket_ids,
                metric={"date": day, "count": int(value), "z_score": round(z_score, 2), "daily_mean": round(mean, 2)},
            )
        )

    if not signals:
        logger.info("未发现单日工单量异常（z 阈值 %.1f）", DAILY_SPIKE_Z_THRESHOLD)
    return signals


def detect_category_growth(df: pd.DataFrame, split_date: str | None = None) -> list[AnomalySignal]:
    """异常 B：某类别工单在时间后半段的日均产出明显上升。

    使用「日均」而非绝对数量，避免两段时间长度不同造成的假增长。
    """
    if df is None or df.empty or "category" not in df.columns:
        return []

    split = split_date or suggest_split_date(df)
    if split is None:
        return []

    rates = period_daily_rates(df, split)
    if rates.empty:
        return []

    subset = rates[rates["维度"] == "问题类型"]
    signals: list[AnomalySignal] = []
    for _, row in subset.iterrows():
        ratio = row["日均变化倍数"]
        early, late = row["前半段日均"], row["后半段日均"]
        if ratio is None or not np.isfinite(ratio):
            continue
        if ratio < CATEGORY_GROWTH_RATIO or row["后半段数量"] < CATEGORY_GROWTH_MIN_COUNT:
            continue

        category = str(row["取值"])
        created = pd.to_datetime(df["created_at"], errors="coerce")
        ids = df[
            (df["category"] == category) & (created >= pd.Timestamp(split))
        ]["ticket_id"].dropna().astype(str).tolist()

        severity: Severity = "关注" if (ratio >= 3 or row["后半段数量"] >= 5) else "提示"
        signals.append(
            AnomalySignal(
                type="类别增长异常",
                description=(
                    f"「{category}」类工单在后半段明显增多，日均由 {early:.2f} 条升至 {late:.2f} 条"
                    f"（约 {ratio:.1f} 倍）。"
                ),
                evidence=(
                    f"以 {split} 为切分点，前半段 {int(row['前半段数量'])} 条 / 后半段 {int(row['后半段数量'])} 条；"
                    f"按日均折算后变化倍数 {ratio:.2f}（阈值 {CATEGORY_GROWTH_RATIO}）。"
                ),
                severity=severity,
                related_tickets=ids,
                metric={
                    "category": category,
                    "split_date": split,
                    "early_daily": early,
                    "late_daily": late,
                    "ratio": ratio,
                },
            )
        )

    if not signals:
        logger.info("未发现类别增长异常（阈值 %.1f 倍）", CATEGORY_GROWTH_RATIO)
    return signals


def detect_long_resolution(df: pd.DataFrame, threshold: float | None = None) -> list[AnomalySignal]:
    """异常 C：处理时长明显偏长的工单（IQR 上界法）。"""
    if df is None or df.empty or "resolution_time_hours" not in df.columns:
        return []

    series = df["resolution_time_hours"].dropna()
    if series.empty:
        return []

    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    limit = threshold if threshold is not None else max(q3 + 1.5 * iqr, LONG_RESOLUTION_MIN_HOURS)

    subset = df[df["resolution_time_hours"] > limit].sort_values(
        "resolution_time_hours", ascending=False
    )
    if subset.empty:
        return []

    ids = subset["ticket_id"].astype(str).tolist()
    detail = "；".join(
        f"{row['ticket_id']}({row['category']}, {row['resolution_time_hours']:g}h, 满意度{row['satisfaction']:g})"
        for _, row in subset.iterrows()
    )
    categories = subset["category"].value_counts().to_dict()
    return [
        AnomalySignal(
            type="处理时长异常",
            description=f"共 {len(subset)} 条工单处理时长超过稳健上界 {limit:.1f} 小时，建议确认是否卡在某个审批或物流环节。",
            evidence=(
                f"IQR 法：Q1={q1:.1f}h，Q3={q3:.1f}h，上界=Q3+1.5×IQR={limit:.1f}h；"
                f"命中工单：{detail}。"
            ),
            severity="关注" if len(subset) >= 3 else "提示",
            related_tickets=ids,
            metric={
                "threshold_hours": round(limit, 2),
                "max_hours": float(subset["resolution_time_hours"].max()),
                "category_distribution": {str(k): int(v) for k, v in categories.items()},
            },
        )
    ]


def detect_low_satisfaction(df: pd.DataFrame, threshold: int = LOW_SATISFACTION_THRESHOLD) -> list[AnomalySignal]:
    """异常 D：低满意度工单，并检查其与处理时长 / 类别 / 优先级 / 渠道的关联。"""
    if df is None or df.empty or "satisfaction" not in df.columns:
        return []

    low = df[df["satisfaction"] <= threshold]
    if low.empty:
        return []

    total = len(df)
    overall_avg = df["satisfaction"].mean()
    top_categories = low["category"].value_counts().head(3).to_dict()
    top_channels = low["channel"].value_counts().to_dict() if "channel" in low else {}
    long_ratio = (
        float((low["resolution_time_hours"] > df["resolution_time_hours"].median()).mean())
        if "resolution_time_hours" in low
        else float("nan")
    )

    evidence_parts = [
        f"满意度 ≤{threshold} 的工单 {len(low)} 条，占全部 {total} 条的 {len(low) / total:.1%}",
        f"全部工单平均满意度 {overall_avg:.2f}",
        f"低满意度工单的问题类型分布：{top_categories}",
    ]
    if top_channels:
        evidence_parts.append(f"渠道分布：{top_channels}")
    if not np.isnan(long_ratio):
        evidence_parts.append(f"其中 {long_ratio:.1%} 的处理时长高于整体中位数")

    return [
        AnomalySignal(
            type="低满意度",
            description=(
                f"检测到 {len(low)} 条低满意度工单（评分 ≤{threshold}），"
                f"其中「{max(top_categories, key=top_categories.get)}」占比最高。"
            ),
            evidence="；".join(evidence_parts) + "。",
            severity="关注" if len(low) / total >= 0.2 else "提示",
            related_tickets=low["ticket_id"].astype(str).tolist(),
            metric={
                "low_count": int(len(low)),
                "low_ratio": round(len(low) / total, 4),
                "avg_satisfaction": round(float(overall_avg), 2),
                "category_distribution": {str(k): int(v) for k, v in top_categories.items()},
                "long_resolution_ratio": None if np.isnan(long_ratio) else round(long_ratio, 4),
            },
        )
    ]


def detect_high_priority_unresolved(df: pd.DataFrame, high_priority: str = "高") -> list[AnomalySignal]:
    """异常 E：高优先级且未解决 —— 主管应直接关注的运营信号。"""
    if df is None or df.empty or "is_resolved" not in df.columns or "priority" not in df.columns:
        return []

    unresolved = df[~df["is_resolved"].fillna(False).astype(bool)]
    subset = unresolved[unresolved["priority"] == high_priority]
    if len(subset) < HIGH_PRIORITY_UNRESOLVED_MIN:
        return []

    subset = subset.sort_values("resolution_time_hours", ascending=False)
    detail = "；".join(
        f"{row['ticket_id']}({row['category']}, 已挂起 {row['resolution_time_hours']:g}h, 满意度{row['satisfaction']:g})"
        for _, row in subset.iterrows()
    )
    hours = subset["resolution_time_hours"].dropna()
    return [
        AnomalySignal(
            type="高优先级未解决",
            description=(
                f"存在 {len(subset)} 条高优先级工单尚未解决，"
                "属于用户侧体感最差的一类，建议优先排期确认。"
            ),
            evidence=(
                f"筛选条件：priority={high_priority} 且 is_resolved=false；"
                f"未解决高优工单占全部未解决工单 {len(subset)}/{len(unresolved)}；"
                f"其中最长已挂起 {float(hours.max()):g} 小时。明细：{detail}。"
            ),
            severity="高",
            related_tickets=subset["ticket_id"].astype(str).tolist(),
            metric={
                "count": int(len(subset)),
                "unresolved_total": int(len(unresolved)),
                "max_hours": None if hours.empty else float(hours.max()),
            },
        )
    ]


def detect_repeat_clusters(df: pd.DataFrame, cluster_table: pd.DataFrame) -> list[AnomalySignal]:
    """异常 F：重复 / 相似问题簇（由 text_analysis 提供的聚类结果生成信号）。"""
    if cluster_table is None or cluster_table.empty:
        return []

    subset = cluster_table[cluster_table["工单数"] >= REPEAT_CLUSTER_MIN_SIZE]
    signals: list[AnomalySignal] = []
    for _, row in subset.iterrows():
        ids = [item.strip() for item in str(row["ticket_ids"]).split(",")]
        signals.append(
            AnomalySignal(
                type="重复问题",
                description=(
                    f"{row['cluster_id']} 簇内 {int(row['工单数'])} 条工单描述高度相似，"
                    "可能存在同一根因或同一类批量问题。"
                ),
                evidence=(
                    f"TF-IDF 字符 1-2gram + 余弦相似度聚类（连通分量）："
                    f"簇内工单 {row['ticket_ids']}；涉及分类「{row['涉及分类']}」；"
                    f"代表描述「{row['代表描述']}」。"
                ),
                severity="关注" if int(row["工单数"]) >= 3 else "提示",
                related_tickets=ids,
                metric={"cluster_id": row["cluster_id"], "size": int(row["工单数"])},
            )
        )
    return signals


def detect_all_anomalies(
    df: pd.DataFrame,
    cluster_table: pd.DataFrame | None = None,
    split_date: str | None = None,
) -> list[AnomalySignal]:
    """执行全部异常检测，返回信号列表（按严重程度排序）。"""
    if df is None or df.empty:
        logger.warning("空数据集，跳过异常检测")
        return []

    signals: list[AnomalySignal] = []
    signals.extend(detect_high_priority_unresolved(df))
    signals.extend(detect_category_growth(df, split_date=split_date))
    signals.extend(detect_daily_volume_anomaly(df))
    signals.extend(detect_long_resolution(df))
    signals.extend(detect_low_satisfaction(df))
    if cluster_table is not None:
        signals.extend(detect_repeat_clusters(df, cluster_table))

    order = {"高": 0, "关注": 1, "提示": 2}
    signals.sort(key=lambda item: order.get(item.severity, 9))
    logger.info("异常检测完成：共 %d 条信号", len(signals))
    return signals


def signals_dataframe(signals: list[AnomalySignal]) -> pd.DataFrame:
    """信号列表转 DataFrame，便于 Dashboard 展示。"""
    columns = ["type", "severity", "description", "evidence", "related_tickets", "related_count"]
    if not signals:
        return pd.DataFrame(columns=columns)
    rows = []
    for signal in signals:
        payload = signal.to_dict()
        payload["related_count"] = len(signal.related_tickets)
        payload["related_tickets"] = ", ".join(signal.related_tickets)
        rows.append(payload)
    return pd.DataFrame(rows)[columns]


def high_priority_unresolved_table(df: pd.DataFrame, high_priority: str = "高") -> pd.DataFrame:
    """高优先级未解决工单明细表（独立函数，便于测试与页面展示）。"""
    columns = ["ticket_id", "created_at", "category", "priority", "channel", "resolution_time_hours", "satisfaction"]
    if df is None or df.empty or "is_resolved" not in df.columns:
        return pd.DataFrame(columns=columns)

    subset = df[~df["is_resolved"].fillna(False).astype(bool)]
    subset = subset[subset["priority"] == high_priority]
    subset = subset.sort_values("resolution_time_hours", ascending=False)
    keep = [c for c in columns if c in subset.columns]
    return subset[keep].reset_index(drop=True)
