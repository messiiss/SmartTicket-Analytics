"""异常检测测试。

重点验证：
1. 每条信号都必须包含 evidence（判断依据）与 severity；
2. 高优先级未解决识别准确；
3. 长时间处理工单识别准确；
4. 阈值参数生效（可调）；
5. 空数据不崩溃。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.anomaly_detection import (
    AnomalySignal,
    detect_all_anomalies,
    detect_category_growth,
    detect_daily_volume_anomaly,
    detect_high_priority_unresolved,
    detect_long_resolution,
    detect_low_satisfaction,
    detect_repeat_clusters,
    high_priority_unresolved_table,
    signals_dataframe,
)
from src.data_cleaner import clean_tickets
from src.data_loader import load_tickets
from src.text_analysis import cluster_similar_tickets


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return clean_tickets(load_tickets().dataframe).dataframe


@pytest.fixture(scope="module")
def clusters(df: pd.DataFrame) -> pd.DataFrame:
    return cluster_similar_tickets(df)


def _assert_signal_contract(signal: AnomalySignal) -> None:
    assert signal.type, "异常类型不能为空"
    assert signal.description, "异常描述不能为空"
    assert signal.evidence, "异常必须有判断依据"
    assert signal.severity in {"高", "关注", "提示"}
    assert isinstance(signal.related_tickets, list)


def test_high_priority_unresolved_detection(df: pd.DataFrame) -> None:
    signals = detect_high_priority_unresolved(df)
    assert signals, "本数据集存在高优先级未解决工单，应被检出"
    signal = signals[0]
    _assert_signal_contract(signal)
    assert signal.type == "高优先级未解决"
    assert signal.severity == "高"

    expected = set(
        df[(df["priority"] == "高") & (~df["is_resolved"].fillna(False).astype(bool))]["ticket_id"]
    )
    assert set(signal.related_tickets) == expected
    assert len(signal.related_tickets) > 0


def test_high_priority_unresolved_table(df: pd.DataFrame) -> None:
    table = high_priority_unresolved_table(df)
    assert not table.empty
    assert (table["priority"] == "高").all()
    ids = set(table["ticket_id"])
    assert ids.isdisjoint(set(df[df["is_resolved"].fillna(False).astype(bool)]["ticket_id"]))


def test_high_priority_unresolved_absent_when_all_resolved(df: pd.DataFrame) -> None:
    all_resolved = df.copy()
    all_resolved["is_resolved"] = True
    assert detect_high_priority_unresolved(all_resolved) == []


def test_long_resolution_detection_uses_iqr_fence(df: pd.DataFrame) -> None:
    signals = detect_long_resolution(df)
    assert signals
    signal = signals[0]
    _assert_signal_contract(signal)
    threshold = signal.metric["threshold_hours"]
    assert threshold > 0
    for ticket_id in signal.related_tickets:
        value = float(df.loc[df["ticket_id"] == ticket_id, "resolution_time_hours"].iloc[0])
        assert value > threshold


def test_long_resolution_threshold_is_configurable(df: pd.DataFrame) -> None:
    strict = detect_long_resolution(df, threshold=200)
    loose = detect_long_resolution(df, threshold=10)
    assert strict == [], "阈值 200 小时时不应有命中"
    assert loose and len(loose[0].related_tickets) > len(detect_long_resolution(df)[0].related_tickets)


def test_low_satisfaction_detection(df: pd.DataFrame) -> None:
    signals = detect_low_satisfaction(df, threshold=2)
    assert signals
    signal = signals[0]
    _assert_signal_contract(signal)
    assert signal.metric["low_count"] == int((df["satisfaction"] <= 2).sum())
    assert 0 < signal.metric["low_ratio"] <= 1


def test_category_growth_detection(df: pd.DataFrame) -> None:
    signals = detect_category_growth(df)
    for signal in signals:
        _assert_signal_contract(signal)
        assert signal.type == "类别增长异常"
        assert signal.metric["ratio"] >= 2.0
        assert signal.metric["late_daily"] > signal.metric["early_daily"]
        category = signal.metric["category"]
        for ticket_id in signal.related_tickets:
            row = df.loc[df["ticket_id"] == ticket_id].iloc[0]
            assert row["category"] == category


def test_daily_volume_anomaly_evidence_contains_stats(df: pd.DataFrame) -> None:
    signals = detect_daily_volume_anomaly(df)
    for signal in signals:
        _assert_signal_contract(signal)
        assert "z-score" in signal.evidence


def test_repeat_clusters_signal_requires_cluster_table(df: pd.DataFrame, clusters: pd.DataFrame) -> None:
    signals = detect_repeat_clusters(df, clusters)
    assert len(signals) == len(clusters)
    for signal in signals:
        _assert_signal_contract(signal)
        assert signal.type == "重复问题"


def test_detect_all_anomalies_sorted_by_severity(df: pd.DataFrame, clusters: pd.DataFrame) -> None:
    signals = detect_all_anomalies(df, cluster_table=clusters)
    assert signals
    order = {"高": 0, "关注": 1, "提示": 2}
    severities = [order[s.severity] for s in signals]
    assert severities == sorted(severities)
    for signal in signals:
        _assert_signal_contract(signal)


def test_signals_dataframe_shape(df: pd.DataFrame, clusters: pd.DataFrame) -> None:
    signals = detect_all_anomalies(df, cluster_table=clusters)
    frame = signals_dataframe(signals)
    assert len(frame) == len(signals)
    assert {"type", "severity", "description", "evidence", "related_tickets"} <= set(frame.columns)


def test_empty_inputs_are_safe() -> None:
    empty = pd.DataFrame()
    assert detect_all_anomalies(empty) == []
    assert detect_high_priority_unresolved(empty) == []
    assert detect_long_resolution(empty) == []
    assert detect_low_satisfaction(empty) == []
    assert detect_category_growth(empty) == []
    assert detect_daily_volume_anomaly(empty) == []
    assert signals_dataframe([]).empty


def test_threshold_change_affects_cluster_count(df: pd.DataFrame) -> None:
    loose = cluster_similar_tickets(df, threshold=0.15)
    strict = cluster_similar_tickets(df, threshold=0.60)
    assert len(loose) >= len(strict)
