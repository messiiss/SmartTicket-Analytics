"""数据加载与字段检查的测试。"""

from __future__ import annotations

import json

import pytest

from src.data_loader import (
    EXPECTED_FIELDS,
    TicketLoadError,
    inspect_fields,
    load_field_documentation,
    load_raw_records,
    load_tickets,
)


def test_load_raw_records_returns_list_of_dicts() -> None:
    records = load_raw_records()
    assert isinstance(records, list)
    assert records, "tickets.json 不应为空"
    assert all(isinstance(item, dict) for item in records)


def test_dataset_size_is_50() -> None:
    """附件数据规模断言：50 条工单。"""
    assert len(load_raw_records()) == 50


def test_dataframe_shape_and_expected_fields() -> None:
    result = load_tickets()
    assert result.row_count == 50
    for field in EXPECTED_FIELDS:
        assert field in result.dataframe.columns, f"缺少字段 {field}"
    assert result.missing_fields == []


def test_ticket_id_is_unique() -> None:
    result = load_tickets()
    ids = result.dataframe["ticket_id"]
    assert ids.notna().all()
    assert ids.is_unique


def test_inspect_fields_detects_missing_and_unexpected() -> None:
    records = [{"ticket_id": "T001", "extra_field": 1}]
    missing, unexpected = inspect_fields(records)
    assert "category" in missing
    assert "extra_field" in unexpected


def test_load_missing_file_raises_readable_error(tmp_path) -> None:
    with pytest.raises(TicketLoadError) as excinfo:
        load_raw_records(tmp_path / "not_exists.json")
    assert "未找到工单数据文件" in str(excinfo.value)


def test_load_invalid_json_raises(tmp_path) -> None:
    broken = tmp_path / "tickets.json"
    broken.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(TicketLoadError):
        load_raw_records(broken)


def test_load_empty_list_raises(tmp_path) -> None:
    empty = tmp_path / "tickets.json"
    empty.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(TicketLoadError):
        load_raw_records(empty)


def test_wrapped_payload_is_supported(tmp_path) -> None:
    """兼容 {"tickets": [...]} 结构。"""
    wrapped = tmp_path / "tickets.json"
    wrapped.write_text(json.dumps({"tickets": [{"ticket_id": "T001"}]}), encoding="utf-8")
    records = load_raw_records(wrapped)
    assert len(records) == 1
    assert records[0]["ticket_id"] == "T001"


def test_missing_optional_field_is_backfilled(tmp_path) -> None:
    """缺少可选字段时应补列而不报错。"""
    payload = [
        {
            "ticket_id": "T001",
            "created_at": "2024-06-01 09:00",
            "category": "支付问题",
            "description": "支付失败",
            "priority": "高",
            "resolution_time_hours": 2,
            "satisfaction": 3,
            "is_resolved": True,
        }
    ]
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = load_tickets(path)
    assert "channel" in result.dataframe.columns
    assert result.dataframe["channel"].isna().all()


def test_field_documentation_is_readable() -> None:
    content = load_field_documentation()
    assert "工单字段说明" in content
    assert "resolution_time_hours" in content


def test_field_documentation_missing_returns_empty(tmp_path) -> None:
    assert load_field_documentation(tmp_path / "none.md") == ""
