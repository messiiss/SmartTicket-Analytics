"""指标计算模块测试。

断言策略：
- 与 pandas 直接计算的结果做交叉验证（防止模块内部算错）；
- 对附件数据集本身的固定事实（如 50 条、8 条未解决）做锚点断言。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_cleaner import clean_tickets
from src.data_loader import load_tickets
from src.metrics import (
    correlation_analysis,
    group_summary,
    high_risk_categories,
    long_tail_tickets,
    low_satisfaction_tickets,
    overall_kpis,
    resolution_stats,
    satisfaction_distribution,
    unresolved_tickets,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return clean_tickets(load_tickets().dataframe).dataframe


def test_total_ticket_count(df: pd.DataFrame) -> None:
    assert overall_kpis(df).total_tickets == 50
    assert len(df) == 50


def test_resolved_and_unresolved_counts(df: pd.DataFrame) -> None:
    kpi = overall_kpis(df)
    assert kpi.resolved == int(df["is_resolved"].fillna(False).astype(bool).sum())
    assert kpi.unresolved == kpi.total_tickets - kpi.resolved
    assert kpi.resolved + kpi.unresolved == 50


def test_average_resolution_hours_matches_pandas(df: pd.DataFrame) -> None:
    kpi = overall_kpis(df)
    expected = round(float(df["resolution_time_hours"].mean()), 2)
    assert kpi.avg_resolution_hours == pytest.approx(expected)
    assert kpi.median_resolution_hours == pytest.approx(
        round(float(df["resolution_time_hours"].median()), 2)
    )


def test_average_satisfaction_matches_pandas(df: pd.DataFrame) -> None:
    kpi = overall_kpis(df)
    assert kpi.avg_satisfaction == pytest.approx(round(float(df["satisfaction"].mean()), 2))


def test_date_range_and_days(df: pd.DataFrame) -> None:
    kpi = overall_kpis(df)
    assert kpi.date_start == "2024-06-01"
    assert kpi.date_end == "2024-06-11"
    assert kpi.days == 11


def test_group_summary_category(df: pd.DataFrame) -> None:
    summary = group_summary(df, "category")
    assert not summary.empty
    assert summary["工单数"].sum() == 50
    assert summary["占比"].sum() == pytest.approx(1.0, abs=0.01)
    # 占比 = 工单数 / 总量
    row = summary.iloc[0]
    assert row["占比"] == pytest.approx(row["工单数"] / 50, abs=0.0001)


def test_group_summary_priority_sorted_by_business_order(df: pd.DataFrame) -> None:
    summary = group_summary(df, "priority")
    assert summary["priority"].tolist() == ["高", "中", "低"]
    assert summary["工单数"].sum() == 50


def test_group_summary_empty_dataframe() -> None:
    summary = group_summary(pd.DataFrame(), "category")
    assert summary.empty
    assert "工单数" in summary.columns


def test_resolution_stats_robustness(df: pd.DataFrame) -> None:
    stats = resolution_stats(df)
    assert stats["count"] == int(df["resolution_time_hours"].notna().sum())
    assert stats["mean"] >= stats["median"], "存在长尾工单时均值应不小于中位数"
    assert stats["iqr_upper_fence"] > stats["q3"]


def test_long_tail_tickets_are_above_fence(df: pd.DataFrame) -> None:
    fence = resolution_stats(df)["iqr_upper_fence"]
    long_tail = long_tail_tickets(df)
    assert not long_tail.empty
    assert (long_tail["resolution_time_hours"] > fence).all()
    assert long_tail["resolution_time_hours"].is_monotonic_decreasing


def test_satisfaction_distribution_sums_to_total(df: pd.DataFrame) -> None:
    distribution = satisfaction_distribution(df)
    assert distribution["工单数"].sum() == int(df["satisfaction"].notna().sum())
    assert distribution["satisfaction"].tolist() == sorted(distribution["satisfaction"].tolist())


def test_low_satisfaction_threshold_respected(df: pd.DataFrame) -> None:
    low = low_satisfaction_tickets(df, threshold=2)
    assert (low["satisfaction"] <= 2).all()
    assert len(low) == int((df["satisfaction"] <= 2).sum())


def test_unresolved_tickets_match_flag(df: pd.DataFrame) -> None:
    unresolved = unresolved_tickets(df)
    assert len(unresolved) == int((~df["is_resolved"].fillna(False).astype(bool)).sum())


def test_correlation_analysis_has_no_certainty_wording(df: pd.DataFrame) -> None:
    table = correlation_analysis(df)
    assert not table.empty
    assert all("因果" in text for text in table["解读"])


def test_high_risk_categories_requires_two_flags(df: pd.DataFrame) -> None:
    table = high_risk_categories(df)
    for _, row in table.iterrows():
        assert len(str(row["命中特征"]).split(" + ")) >= 2


def test_empty_dataframe_does_not_crash() -> None:
    empty = pd.DataFrame(columns=["ticket_id", "category", "priority", "channel", "satisfaction", "resolution_time_hours", "is_resolved"])
    kpi = overall_kpis(empty)
    assert kpi.total_tickets == 0
    assert kpi.avg_satisfaction is None
    assert kpi.date_start is None
    assert resolution_stats(empty)["count"] == 0
    assert correlation_analysis(empty).empty
