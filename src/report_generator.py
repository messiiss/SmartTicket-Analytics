"""报告生成模块：把已计算好的结构化指标渲染成「主管摘要」与 Markdown 报告。

重要约束：
- 本模块**只做模板化渲染**，所有数字都来自上游函数的计算结果。
- 不调用任何外部 API，不生成任何未经计算的数字。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .anomaly_detection import AnomalySignal
from .data_cleaner import PRIORITY_ORDER
from .metrics import OverallKPI
from .trend_analysis import TrendSummary

logger = logging.getLogger(__name__)

SEVERITY_MARK: dict[str, str] = {"高": "【高】", "关注": "【关注】", "提示": "【提示】"}


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    """统一的数字格式化：None 显示为 '暂无数据'，避免出现 'None' 字样。"""
    if value is None:
        return "暂无数据"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{round(value, digits)}{suffix}"


def generate_manager_summary(
    kpi: OverallKPI,
    trend: TrendSummary,
    category_summary: pd.DataFrame,
    anomaly_signals: list[AnomalySignal],
    repeat_cluster_count: int = 0,
    top_n: int = 3,
) -> str:
    """生成主管摘要（纯模板 + 真实数字）。"""
    lines: list[str] = []

    lines.append("【总体情况】")
    lines.append(
        f"本周期（{kpi.date_start or '暂无数据'} 至 {kpi.date_end or '暂无数据'}，共 {kpi.days} 天）"
        f"共收到 {kpi.total_tickets} 条工单，已解决 {kpi.resolved} 条，未解决 {kpi.unresolved} 条"
        f"（未解决率 {kpi.unresolved_rate:.1%}）。"
        f"平均处理时长 {_fmt(kpi.avg_resolution_hours, suffix=' 小时')}，"
        f"中位处理时长 {_fmt(kpi.median_resolution_hours, suffix=' 小时')}，"
        f"平均满意度 {_fmt(kpi.avg_satisfaction)}。"
    )

    lines.append("")
    lines.append("【趋势】")
    if trend.first_half_daily_avg is None or trend.second_half_daily_avg is None:
        lines.append("数据时间跨度不足，暂无法给出趋势判断。")
    else:
        direction = "上升" if (trend.change_ratio or 0) > 0 else ("下降" if (trend.change_ratio or 0) < 0 else "基本持平")
        lines.append(
            f"周期内日均工单量 {_fmt(trend.daily_average)} 条/天，"
            f"峰值出现在 {trend.peak_date}（{trend.peak_count} 条）。"
            f"按时间前后半段对比，日均工单量呈{direction}趋势："
            f"前半段 {_fmt(trend.first_half_daily_avg)} 条/天 → 后半段 {_fmt(trend.second_half_daily_avg)} 条/天"
            f"（变化 {_fmt((trend.change_ratio or 0) * 100, 1, '%')}）。"
        )
        if trend.weekly_available:
            lines.append("数据跨度已满足周趋势分析条件。")
        else:
            lines.append(f"说明：本周期仅 {trend.days} 天，不足两周，因此未做周维度趋势，改用前后半段日均对比。")

    lines.append("")
    lines.append("【重点问题】")
    if category_summary is None or category_summary.empty:
        lines.append("暂无可用的分类统计结果。")
    else:
        ranked = category_summary.sort_values("工单数", ascending=False).head(top_n)
        parts = [
            f"{row['category']} {int(row['工单数'])} 条（占 {row['占比']:.1%}，"
            f"平均处理 {_fmt(row['平均处理时长(h)'])} 小时，平均满意度 {_fmt(row['平均满意度'])}）"
            for _, row in ranked.iterrows()
        ]
        lines.append("工单量最高的前三类问题为：" + "；".join(parts) + "。")

        slowest = category_summary.dropna(subset=["平均处理时长(h)"]).sort_values(
            "平均处理时长(h)", ascending=False
        )
        if not slowest.empty:
            row = slowest.iloc[0]
            lines.append(
                f"平均处理时长最长的是「{row['category']}」（{_fmt(row['平均处理时长(h)'])} 小时，"
                f"样本 {int(row['工单数'])} 条），建议确认该类是否存在审批或跨部门协作瓶颈。"
            )

        lowest = category_summary.dropna(subset=["平均满意度"]).sort_values("平均满意度")
        if not lowest.empty:
            row = lowest.iloc[0]
            lines.append(
                f"平均满意度最低的是「{row['category']}」（{_fmt(row['平均满意度'])}，"
                f"样本 {int(row['工单数'])} 条），建议抽取具体工单了解体验卡点。"
            )

    lines.append("")
    lines.append("【异常信号】")
    if not anomaly_signals:
        lines.append("本期未检测到需要特别关注的异常信号。")
    else:
        by_severity: dict[str, int] = {}
        for signal in anomaly_signals:
            by_severity[signal.severity] = by_severity.get(signal.severity, 0) + 1
        breakdown = "，".join(f"{key} {value} 条" for key, value in by_severity.items())
        lines.append(f"共检测到 {len(anomaly_signals)} 个异常信号（{breakdown}）：")
        for index, signal in enumerate(anomaly_signals, start=1):
            lines.append(
                f"{index}. {SEVERITY_MARK.get(signal.severity, '')}{signal.type}："
                f"{signal.description}"
            )
        if repeat_cluster_count:
            lines.append(f"其中相似/重复问题簇 {repeat_cluster_count} 组，见「相似工单」页面。")

    lines.append("")
    lines.append("【建议关注】")
    suggestions: list[str] = []
    for signal in anomaly_signals:
        if signal.severity != "高":
            continue
        suggestions.append(f"{signal.type}：涉及 {', '.join(signal.related_tickets)}，建议当日确认进展。")
    growth = [s for s in anomaly_signals if s.type == "类别增长异常"]
    for signal in growth:
        category = signal.metric.get("category", "该类别")
        suggestions.append(
            f"「{category}」类工单增长较快（日均变化 {signal.metric.get('ratio')} 倍），"
            "建议核对是否由同一产品/流程问题引起。"
        )
    long_time = [s for s in anomaly_signals if s.type == "处理时长异常"]
    for signal in long_time:
        suggestions.append(
            f"处理时长超过 {signal.metric.get('threshold_hours')} 小时的工单建议逐条复盘，"
            "确认卡点发生在客服、仓储还是物流环节。"
        )
    if repeat_cluster_count:
        suggestions.append("相似问题簇建议合并处理：同一根因批量回复可显著降低重复工单量。")
    if not suggestions:
        suggestions.append("本期无高风险事项，保持现有节奏并持续观察指标变化。")

    lines.extend(f"- {item}" for item in suggestions)
    lines.append("")
    lines.append("说明：以上信号基于当前数据的统计偏离给出，用于提示关注方向，不等同于已确认的业务故障。")
    return "\n".join(lines)


def build_markdown_report(
    kpi: OverallKPI,
    trend: TrendSummary,
    summary_text: str,
    tables: dict[str, pd.DataFrame],
    anomaly_signals: list[AnomalySignal],
    quality_summary: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> str:
    """生成 Markdown 版分析报告。"""
    timestamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blocks: list[str] = [
        "# SmartTicket Analytics 分析报告",
        "",
        f"> 生成时间：{timestamp}　|　数据来源：`data/tickets.json`（{kpi.total_tickets} 条工单）",
        "",
        "## 一、主管摘要",
        "",
        "```text",
        summary_text,
        "```",
        "",
        "## 二、核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 工单总数 | {kpi.total_tickets} |",
        f"| 已解决 | {kpi.resolved} |",
        f"| 未解决 | {kpi.unresolved} |",
        f"| 未解决率 | {kpi.unresolved_rate:.1%} |",
        f"| 平均处理时长 | {_fmt(kpi.avg_resolution_hours, suffix=' 小时')} |",
        f"| 中位处理时长 | {_fmt(kpi.median_resolution_hours, suffix=' 小时')} |",
        f"| 最长处理时长 | {_fmt(kpi.max_resolution_hours, suffix=' 小时')} |",
        f"| 平均满意度 | {_fmt(kpi.avg_satisfaction)} |",
        f"| 高优先级工单 | {kpi.high_priority_tickets} |",
        f"| 数据范围 | {kpi.date_start} ~ {kpi.date_end}（{kpi.days} 天） |",
        "",
    ]

    for title, table in tables.items():
        if table is None or table.empty:
            continue
        blocks.extend([f"## {title}", "", _to_markdown_table(table), ""])

    blocks.extend(["## 异常信号明细", ""])
    if not anomaly_signals:
        blocks.append("本期未检测到异常信号。")
    else:
        for index, signal in enumerate(anomaly_signals, start=1):
            blocks.extend(
                [
                    f"### {index}. {SEVERITY_MARK.get(signal.severity, '')}{signal.type}",
                    "",
                    f"- **描述**：{signal.description}",
                    f"- **判断依据**：{signal.evidence}",
                    f"- **严重程度**：{signal.severity}",
                    f"- **关联工单**：{', '.join(signal.related_tickets) if signal.related_tickets else '无'}",
                    "",
                ]
            )

    if quality_summary:
        blocks.extend(
            [
                "## 数据质量",
                "",
                f"- 原始记录数：{quality_summary.get('raw_rows')}",
                f"- 清洗后记录数：{quality_summary.get('clean_rows')}",
                f"- error 级问题：{quality_summary.get('error_count')}　warning 级问题：{quality_summary.get('warning_count')}",
                f"- 缺失单元格：{quality_summary.get('missing_cells')}",
                f"- 重复工单编号：{quality_summary.get('duplicate_ticket_ids')}",
                f"- 数据时间范围：{quality_summary.get('date_start')} ~ {quality_summary.get('date_end')}",
                "",
            ]
        )

    blocks.extend(
        [
            "---",
            "",
            "本报告所有数字均由程序基于 `data/tickets.json` 实时计算得出，未做任何人工填写或推测。",
        ]
    )
    return "\n".join(blocks)


def _to_markdown_table(dataframe: pd.DataFrame) -> str:
    """把 DataFrame 渲染为 Markdown 表格（不依赖 tabulate）。"""
    headers = [str(column) for column in dataframe.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in dataframe.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{round(value, 4)}")
            elif pd.isna(value):
                cells.append("-")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_report(content: str, path: str | Path) -> Path:
    """保存报告到文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    logger.info("报告已写入：%s", target)
    return target


def priority_order_map() -> dict[str, int]:
    """暴露优先级排序权重，供其它模块复用。"""
    return dict(PRIORITY_ORDER)
