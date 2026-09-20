"""SmartTicket Analytics —— Streamlit Dashboard 入口。

运行方式::

    streamlit run app.py

页面通过左侧 sidebar 切换，也支持 URL 参数直接打开指定页，便于截图与分享：
``http://localhost:8501/?page=Anomaly%20Detection``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import (  # noqa: E402
    detect_all_anomalies,
    high_priority_unresolved_table,
    signals_dataframe,
)
from src.data_cleaner import clean_tickets, summarize_data_quality  # noqa: E402
from src.data_loader import load_field_documentation, load_tickets  # noqa: E402
from src.metrics import (  # noqa: E402
    correlation_analysis,
    group_summary,
    high_risk_categories,
    long_tail_tickets,
    low_satisfaction_tickets,
    overall_kpis,
    satisfaction_by_dimension,
    satisfaction_distribution,
    unresolved_tickets,
)
from src.report_generator import generate_manager_summary  # noqa: E402
from src.text_analysis import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    cluster_similar_tickets,
    find_similar_pairs,
    top_keywords,
)
from src.trend_analysis import (  # noqa: E402
    daily_counts,
    period_daily_rates,
    suggest_split_date,
    trend_summary,
    weekly_counts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

FONT_FAMILY = "Microsoft YaHei, PingFang SC, Noto Sans CJK SC, Hiragino Sans GB, sans-serif"
PALETTE = ["#2E5AAC", "#1D9E75", "#BA7517", "#D4537E", "#7F77DD", "#888780"]

PAGES = [
    "Overview",
    "Trend Analysis",
    "Category Analysis",
    "Priority Analysis",
    "Resolution Analysis",
    "Satisfaction Analysis",
    "Anomaly Detection",
    "Similar Tickets",
]

st.set_page_config(
    page_title="SmartTicket Analytics · 客服工单趋势与异常分析",
    page_icon="📊",
    layout="wide",
)


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """统一图表样式：字体、留白、hover。"""
    fig.update_layout(
        height=height,
        font=dict(family=FONT_FAMILY, size=13, color="#2C2C2A"),
        margin=dict(l=40, r=24, t=56, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D3D1C7")
    fig.update_yaxes(gridcolor="#F1EFE8", zeroline=False)
    return fig


@st.cache_data(show_spinner="正在加载并清洗工单数据…")
def load_data() -> dict[str, Any]:
    """加载 → 清洗 → 计算全部指标（结果缓存，指纹以文件修改时间为准）。"""
    load_result = load_tickets()
    clean_result = clean_tickets(load_result.dataframe)
    df = clean_result.dataframe

    kpi = overall_kpis(df)
    trend = trend_summary(df)
    split = suggest_split_date(df)
    clusters = cluster_similar_tickets(df, threshold=DEFAULT_SIMILARITY_THRESHOLD)
    anomalies = detect_all_anomalies(df, cluster_table=clusters, split_date=split)

    return {
        "df": df,
        "kpi": kpi,
        "trend": trend,
        "split": split,
        "quality": summarize_data_quality(clean_result),
        "quality_issues": clean_result.issues_dataframe(),
        "clusters": clusters,
        "anomalies": anomalies,
        "summary": generate_manager_summary(
            kpi=kpi,
            trend=trend,
            category_summary=group_summary(df, "category"),
            anomaly_signals=anomalies,
            repeat_cluster_count=len(clusters),
        ),
        "field_doc": load_field_documentation(),
        "unexpected_fields": load_result.unexpected_fields,
        "missing_fields": load_result.missing_fields,
    }


def metric_row(items: list[tuple[str, str]]) -> None:
    """渲染一行 KPI 卡片。"""
    columns = st.columns(len(items), gap="small")
    for column, (label, value) in zip(columns, items):
        with column:
            st.metric(label=label, value=value)


def page_overview(data: dict[str, Any]) -> None:
    kpi = data["kpi"]
    anomalies = data["anomalies"]
    trend = data["trend"]

    st.subheader("客服工单趋势与异常分析")
    st.caption(
        f"数据范围：{kpi.date_start} ~ {kpi.date_end}（{kpi.days} 天）　|　"
        f"数据源：data/tickets.json　|　异常信号判定阈值均为经验参数，可在「异常检测」页调整"
    )

    metric_row(
        [
            ("总工单数", f"{kpi.total_tickets}"),
            ("已解决", f"{kpi.resolved}"),
            ("未解决", f"{kpi.unresolved}"),
            ("平均处理时长", f"{kpi.avg_resolution_hours} h"),
            ("平均满意度", f"{kpi.avg_satisfaction}"),
            ("异常信号", f"{len(anomalies)}"),
        ]
    )
    metric_row(
        [
            ("未解决率", f"{kpi.unresolved_rate:.1%}"),
            ("中位处理时长", f"{kpi.median_resolution_hours} h"),
            ("最长处理时长", f"{kpi.max_resolution_hours} h"),
            ("高优先级工单", f"{kpi.high_priority_tickets} 条"),
            ("日均工单量", f"{trend.daily_average} 条/天"),
            ("峰值日", f"{trend.peak_date[5:] if trend.peak_date else '-'}（{trend.peak_count} 条）"),
        ]
    )

    st.divider()
    st.markdown("#### 一、主管关注")

    if not anomalies:
        st.success("本期未检测到需要特别关注的异常信号。")
    else:
        severity_to_render = {"高": st.error, "关注": st.warning, "提示": st.info}
        for signal in anomalies:
            render = severity_to_render.get(signal.severity, st.info)
            tickets = "、".join(signal.related_tickets[:8])
            more = "" if len(signal.related_tickets) <= 8 else f" 等 {len(signal.related_tickets)} 条"
            render(
                f"**【{signal.severity}】{signal.type}**　{signal.description}\n\n"
                f"判断依据：{signal.evidence}\n\n"
                f"关联工单：{tickets}{more}"
            )
        st.caption("以上为统计意义上的异常信号，用于提示关注方向，不等同于已确认的业务故障，建议进一步确认。")

    st.divider()
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("#### 二、每日工单量")
        daily = daily_counts(df=data["df"])
        if daily.empty:
            st.info("暂无可用的时间序列数据。")
        else:
            fig = px.line(daily, x="date", y="count", markers=True, color_discrete_sequence=[PALETTE[0]])
            fig.update_traces(
                hovertemplate="日期 %{x|%Y-%m-%d}<br>工单量 %{y} 条<extra></extra>",
                line=dict(width=2.5),
            )
            fig.update_layout(title="每日工单量趋势", xaxis_title="日期", yaxis_title="工单量（条）")
            st.plotly_chart(style_figure(fig), width="stretch")
    with right:
        st.markdown("#### 三、问题类型分布")
        category_summary = group_summary(data["df"], "category")
        if category_summary.empty:
            st.info("暂无分类数据。")
        else:
            fig = px.bar(
                category_summary.sort_values("工单数"),
                x="工单数",
                y="category",
                orientation="h",
                text="工单数",
                color_discrete_sequence=[PALETTE[1]],
            )
            fig.update_traces(
                hovertemplate="%{y}<br>工单数 %{x} 条<extra></extra>", textposition="outside"
            )
            fig.update_layout(title="各问题类型工单数量", xaxis_title="工单数（条）", yaxis_title="")
            st.plotly_chart(style_figure(fig), width="stretch")

    st.markdown("#### 四、自动生成的主管摘要")
    st.text(data["summary"])

    with st.expander("数据质量检查结果 / 字段说明"):
        quality = data["quality"]
        st.write(
            f"原始记录 {quality['raw_rows']} 条，清洗后 {quality['clean_rows']} 条；"
            f"error 级问题 {quality['error_count']} 条，warning 级问题 {quality['warning_count']} 条；"
            f"缺失单元格 {quality['missing_cells']} 个；重复工单编号 {quality['duplicate_ticket_ids']} 个。"
        )
        issues = data["quality_issues"]
        if issues.empty:
            st.success("未发现数据质量问题。")
        else:
            st.dataframe(issues, width="stretch", hide_index=True)
        if data["field_doc"]:
            st.markdown(data["field_doc"])


def page_trend(data: dict[str, Any]) -> None:
    df = data["df"]
    trend = data["trend"]
    st.subheader("Trend Analysis · 时间趋势")
    st.caption("用于判断近期客服压力是否增加、是否出现突然增长。")

    daily = daily_counts(df)
    high_priority_daily = daily_counts(df, priority="高")
    if daily.empty:
        st.info("暂无可用的时间序列数据。")
        return

    metric_row(
        [
            ("覆盖天数", f"{trend.days} 天"),
            ("日均工单量", f"{trend.daily_average} 条/天"),
            ("前半段日均", f"{trend.first_half_daily_avg} 条/天"),
            ("后半段日均", f"{trend.second_half_daily_avg} 条/天"),
        ]
    )

    fig = px.line(daily, x="date", y="count", markers=True, color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(
        hovertemplate="日期 %{x|%Y-%m-%d}<br>工单量 %{y} 条<extra></extra>", line=dict(width=2.5)
    )
    fig.update_layout(title="每日工单量", xaxis_title="日期", yaxis_title="工单量（条）")
    st.plotly_chart(style_figure(fig), width="stretch")

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        if high_priority_daily.empty:
            st.info("无高优先级工单数据。")
        else:
            fig = px.bar(
                high_priority_daily,
                x="date",
                y="count",
                color_discrete_sequence=[PALETTE[2]],
            )
            fig.update_traces(
                hovertemplate="日期 %{x|%Y-%m-%d}<br>高优先级 %{y} 条<extra></extra>"
            )
            fig.update_layout(title="每日高优先级工单量", xaxis_title="日期", yaxis_title="工单量（条）")
            st.plotly_chart(style_figure(fig, 340), width="stretch")
    with col_right:
        st.markdown("#### 前后半段日均对比")
        rates = period_daily_rates(df, data["split"] or "")
        if rates.empty:
            st.info("数据不足，无法做前后半段对比。")
        else:
            st.dataframe(
                rates.sort_values("日均变化倍数", ascending=False),
                width="stretch",
                hide_index=True,
            )
        weekly = weekly_counts(df)
        if weekly.empty:
            st.caption(
                f"说明：本周期仅 {trend.days} 天，不足两周，未做周维度趋势分析，"
                "改用前后半段日均对比以避免周维度样本过少导致的误导。"
            )
        else:
            st.dataframe(weekly, width="stretch", hide_index=True)


def page_category(data: dict[str, Any]) -> None:
    df = data["df"]
    st.subheader("Category Analysis · 问题类型分析")
    summary = group_summary(df, "category")
    if summary.empty:
        st.info("暂无分类数据。")
        return

    fig = px.bar(
        summary.sort_values("工单数", ascending=False),
        x="category",
        y="工单数",
        text="工单数",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.update_traces(hovertemplate="%{x}<br>工单数 %{y} 条<extra></extra>", textposition="outside")
    fig.update_layout(title="各问题类型工单数量", xaxis_title="问题类型", yaxis_title="工单数（条）")
    st.plotly_chart(style_figure(fig), width="stretch")

    st.markdown("#### 分类明细（数量 / 占比 / 处理时长 / 满意度 / 未解决）")
    st.dataframe(summary, width="stretch", hide_index=True)

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        fig = px.bar(
            summary.dropna(subset=["平均处理时长(h)"]).sort_values("平均处理时长(h)"),
            x="平均处理时长(h)",
            y="category",
            orientation="h",
            text="平均处理时长(h)",
            color_discrete_sequence=[PALETTE[2]],
        )
        fig.update_traces(hovertemplate="%{y}<br>平均处理 %{x} 小时<extra></extra>", textposition="outside")
        fig.update_layout(title="各问题类型平均处理时长", xaxis_title="小时", yaxis_title="")
        st.plotly_chart(style_figure(fig, 340), width="stretch")
    with col_right:
        fig = px.bar(
            summary.dropna(subset=["平均满意度"]).sort_values("平均满意度"),
            x="平均满意度",
            y="category",
            orientation="h",
            text="平均满意度",
            color_discrete_sequence=[PALETTE[3]],
        )
        fig.update_traces(hovertemplate="%{y}<br>平均满意度 %{x}<extra></extra>", textposition="outside")
        fig.update_layout(title="各问题类型平均满意度", xaxis_title="满意度（1-5）", yaxis_title="")
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    st.markdown("#### 同时满足「数量多 + 处理慢 + 满意度低」中至少两项的类别")
    risk = high_risk_categories(df)
    if risk.empty:
        st.info("没有类别同时命中两项以上特征。")
    else:
        st.dataframe(risk, width="stretch", hide_index=True)


def page_priority(data: dict[str, Any]) -> None:
    df = data["df"]
    st.subheader("Priority Analysis · 优先级分析")
    summary = group_summary(df, "priority")
    if summary.empty:
        st.info("暂无优先级数据。")
        return

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        counts = df["priority"].value_counts().reset_index()
        counts.columns = ["priority", "count"]
        fig = px.pie(
            counts,
            names="priority",
            values="count",
            hole=0.45,
            color_discrete_sequence=[PALETTE[2], PALETTE[0], PALETTE[1]],
        )
        fig.update_traces(hovertemplate="%{label}<br>%{value} 条（%{percent}）<extra></extra>")
        fig.update_layout(title="优先级分布")
        st.plotly_chart(style_figure(fig, 360), width="stretch")
    with col_right:
        fig = px.bar(
            summary,
            x="priority",
            y="未解决数",
            text="未解决数",
            color_discrete_sequence=[PALETTE[4]],
        )
        fig.update_traces(hovertemplate="%{x}<br>未解决 %{y} 条<extra></extra>", textposition="outside")
        fig.update_layout(title="各优先级未解决工单数", xaxis_title="优先级", yaxis_title="未解决（条）")
        st.plotly_chart(style_figure(fig, 360), width="stretch")

    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown("#### 高优先级工单的每日分布（观察是否集中增长）")
    high_daily = daily_counts(df, priority="高")
    if high_daily.empty:
        st.info("无高优先级工单。")
    else:
        fig = px.bar(high_daily, x="date", y="count", color_discrete_sequence=[PALETTE[2]])
        fig.update_traces(hovertemplate="日期 %{x|%Y-%m-%d}<br>高优先级 %{y} 条<extra></extra>")
        fig.update_layout(title="每日高优先级工单量", xaxis_title="日期", yaxis_title="工单量（条）")
        st.plotly_chart(style_figure(fig, 340), width="stretch")


def page_resolution(data: dict[str, Any]) -> None:
    df = data["df"]
    st.subheader("Resolution Analysis · 处理效率")
    st.caption("同时展示均值与中位数：少量极端值会显著拉高平均值，仅看均值容易误判。")

    long_tail = long_tail_tickets(df)
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        series = df["resolution_time_hours"].dropna()
        if series.empty:
            st.info("暂无处理时长数据。")
        else:
            fig = px.histogram(series, nbins=15, color_discrete_sequence=[PALETTE[1]])
            fig.update_traces(hovertemplate="处理时长 %{x} 小时<br>工单数 %{y}<extra></extra>")
            fig.update_layout(
                title="处理时长分布",
                xaxis_title="处理时长（小时）",
                yaxis_title="工单数（条）",
                showlegend=False,
            )
            st.plotly_chart(style_figure(fig, 340), width="stretch")
    with col_right:
        st.markdown("#### 处理时长关键统计")
        stats = {
            "工单数": int(series.size),
            "平均值": df["resolution_time_hours"].mean().round(2),
            "中位数": df["resolution_time_hours"].median().round(2),
            "最大值": df["resolution_time_hours"].max(),
            "P90": series.quantile(0.9).round(2),
            "IQR 上界": long_tail["resolution_time_hours"].min() if not long_tail.empty else None,
        }
        st.dataframe(
            pd.DataFrame({"统计项": list(stats), "数值": list(stats.values())}),
            width="stretch",
            hide_index=True,
        )
        by_category = group_summary(df, "category")[["category", "平均处理时长(h)", "中位处理时长(h)", "工单数"]]
        fig = px.bar(
            by_category.dropna(subset=["平均处理时长(h)"]),
            x="category",
            y=["平均处理时长(h)", "中位处理时长(h)"],
            barmode="group",
            color_discrete_sequence=[PALETTE[2], PALETTE[0]],
        )
        fig.update_layout(
            title="各问题类型：平均 vs 中位处理时长",
            xaxis_title="问题类型",
            yaxis_title="小时",
            legend_title="",
        )
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    st.markdown("#### 长时间处理工单（超过 IQR 稳健上界）")
    if long_tail.empty:
        st.info("没有超过稳健上界的工单。")
    else:
        st.dataframe(long_tail, width="stretch", hide_index=True)

    st.markdown("#### 未解决工单列表")
    unresolved = unresolved_tickets(df)
    if unresolved.empty:
        st.success("所有工单均已解决。")
    else:
        st.dataframe(unresolved, width="stretch", hide_index=True)


def page_satisfaction(data: dict[str, Any]) -> None:
    df = data["df"]
    st.subheader("Satisfaction Analysis · 满意度分析")

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        distribution = satisfaction_distribution(df)
        if distribution.empty:
            st.info("暂无满意度数据。")
        else:
            fig = px.bar(
                distribution,
                x="satisfaction",
                y="工单数",
                text="工单数",
                color_discrete_sequence=[PALETTE[3]],
            )
            fig.update_traces(
                hovertemplate="评分 %{x}<br>工单数 %{y}（占 %{customdata:.1%}）<extra></extra>",
                textposition="outside",
                customdata=distribution["占比"],
            )
            fig.update_layout(
                title="满意度评分分布",
                xaxis_title="满意度评分（1-5）",
                yaxis_title="工单数（条）",
            )
            st.plotly_chart(style_figure(fig, 340), width="stretch")
    with col_right:
        pivot = (
            df.pivot_table(
                index="category",
                columns="satisfaction",
                values="ticket_id",
                aggfunc="count",
                fill_value=0,
            )
            .sort_index()
        )
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.to_numpy(),
                x=[f"{int(c)} 分" for c in pivot.columns],
                y=[str(i) for i in pivot.index],
                colorscale=[[0, "#F7F6F2"], [1, "#2E5AAC"]],
                text=pivot.to_numpy(),
                texttemplate="%{text}",
                hovertemplate="%{y} · %{x}<br>工单数 %{z}<extra></extra>",
                showscale=False,
            )
        )
        fig.update_layout(
            title="问题类型 × 满意度交叉分布",
            xaxis_title="满意度评分",
            yaxis_title="问题类型",
        )
        st.plotly_chart(style_figure(fig, 340), width="stretch")

    st.markdown("#### 各维度低满意度率（评分 ≤2 视为低满意度）")
    dimension = st.selectbox("选择维度", ["category", "priority", "channel"], index=0)
    cross = satisfaction_by_dimension(df, dimension)
    if cross.empty:
        st.info("暂无数据。")
    else:
        fig = px.bar(
            cross.sort_values("低满意度率"),
            x="低满意度率",
            y=dimension,
            orientation="h",
            text="低满意度率",
            color_discrete_sequence=[PALETTE[4]],
        )
        fig.update_traces(
            texttemplate="%{text:.0%}",
            hovertemplate="%{y}<br>低满意度率 %{x:.1%}<extra></extra>",
            textposition="outside",
        )
        fig.update_layout(title=f"各{dimension} 低满意度率", xaxis_title="低满意度率", yaxis_title="")
        st.plotly_chart(style_figure(fig, 340), width="stretch")
        st.dataframe(cross, width="stretch", hide_index=True)

    st.markdown("#### 低满意度工单明细")
    low = low_satisfaction_tickets(df)
    if low.empty:
        st.success("没有低满意度工单。")
    else:
        st.dataframe(low, width="stretch", hide_index=True)

    st.markdown("#### 维度相关性（Spearman，仅表示相关，不代表因果）")
    st.dataframe(correlation_analysis(df), width="stretch", hide_index=True)


def page_anomaly(data: dict[str, Any]) -> None:
    df = data["df"]
    st.subheader("Anomaly Detection · 异常检测")
    st.caption(
        "每条信号都附带判断依据（evidence）。由于样本量仅 50 条，"
        "这里的结论是「异常信号」而非已确认的业务故障，建议进一步确认。"
    )

    threshold = st.slider(
        "相似问题判定阈值（用于「重复问题」信号，经验参数，可按数据调整）",
        min_value=0.15,
        max_value=0.60,
        value=float(DEFAULT_SIMILARITY_THRESHOLD),
        step=0.05,
    )
    clusters = cluster_similar_tickets(df, threshold=threshold)
    anomalies = detect_all_anomalies(df, cluster_table=clusters, split_date=data["split"])

    severity_counts: dict[str, int] = {"高": 0, "关注": 0, "提示": 0}
    for signal in anomalies:
        severity_counts[signal.severity] = severity_counts.get(signal.severity, 0) + 1
    metric_row(
        [
            ("异常信号总数", f"{len(anomalies)}"),
            ("高", f"{severity_counts.get('高', 0)}"),
            ("关注", f"{severity_counts.get('关注', 0)}"),
            ("提示", f"{severity_counts.get('提示', 0)}"),
            ("相似问题簇", f"{len(clusters)}"),
        ]
    )

    st.markdown("#### 信号明细（含判断依据）")
    for signal in anomalies:
        with st.container(border=True):
            st.markdown(f"**【{signal.severity}】{signal.type} · {signal.description}**")
            st.markdown(f"判断依据：{signal.evidence}")
            st.markdown(
                f"关联工单（{len(signal.related_tickets)} 条）："
                f"{'、'.join(signal.related_tickets) if signal.related_tickets else '无'}"
            )

    st.markdown("#### 高优先级未解决工单")
    high_unresolved = high_priority_unresolved_table(df)
    if high_unresolved.empty:
        st.success("没有高优先级未解决工单。")
    else:
        st.dataframe(high_unresolved, width="stretch", hide_index=True)

    st.markdown("#### 各类异常数量一览")
    frame = signals_dataframe(anomalies)
    if frame.empty:
        st.info("暂无异常信号。")
    else:
        type_counts = (
            frame.groupby(["type", "severity"]).size().reset_index(name="数量")
        )
        fig = px.bar(
            type_counts,
            x="type",
            y="数量",
            color="severity",
            barmode="stack",
            color_discrete_sequence=[PALETTE[2], PALETTE[0], PALETTE[5]],
        )
        fig.update_layout(
            title="异常信号类型 × 严重程度",
            xaxis_title="异常类型",
            yaxis_title="信号数量",
            legend_title="严重程度",
        )
        st.plotly_chart(style_figure(fig, 360), width="stretch")

    with st.expander("查看结构化异常信号 JSON（可直接用于对接其它系统）"):
        st.json([signal.to_dict() for signal in anomalies])


def page_similar(data: dict[str, Any]) -> None:
    df = data["df"]
    st.subheader("Similar Tickets · 相似 / 重复问题")
    st.caption(
        "方法：中文按字符 1-2gram 构造 TF-IDF 向量 → 余弦相似度 → 连通分量聚类。"
        "不引入额外分词依赖，短文本场景下更稳定。"
    )

    threshold = st.slider(
        "相似度阈值（经验参数，可根据业务容忍度调整）",
        min_value=0.15,
        max_value=0.60,
        value=float(DEFAULT_SIMILARITY_THRESHOLD),
        step=0.05,
    )

    pairs = find_similar_pairs(df, threshold=threshold)
    clusters = cluster_similar_tickets(df, threshold=threshold)

    metric_row(
        [
            ("相似工单对", f"{len(pairs)} 组"),
            ("相似问题簇", f"{len(clusters)} 组"),
            ("涉及工单", f"{len(set(pairs['ticket_a']) | set(pairs['ticket_b'])) if not pairs.empty else 0} 条"),
        ]
    )

    if not pairs.empty:
        fig = px.bar(
            pairs.head(12).assign(pair=lambda d: d["ticket_a"] + " ↔ " + d["ticket_b"]),
            x="similarity",
            y="pair",
            orientation="h",
            text="similarity",
            color_discrete_sequence=[PALETTE[0]],
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            hovertemplate="%{y}<br>相似度 %{x:.3f}<extra></extra>",
            textposition="outside",
        )
        fig.update_layout(title="相似度最高的工单对", xaxis_title="余弦相似度", yaxis_title="")
        st.plotly_chart(style_figure(fig, 420), width="stretch")

    st.markdown("#### 相似问题簇（同一根因的批量工单）")
    if clusters.empty:
        st.info(f"当前阈值 {threshold} 下未发现相似问题簇，可适当调低阈值。")
    else:
        st.dataframe(clusters, width="stretch", hide_index=True)

    st.markdown("#### 相似工单对明细")
    if pairs.empty:
        st.info("无相似工单对。")
    else:
        st.dataframe(pairs, width="stretch", hide_index=True)

    with st.expander("高频关键词（TF-IDF 权重聚合，用于观察集中问题）"):
        st.dataframe(top_keywords(df), width="stretch", hide_index=True)


def _initial_page_index() -> int:
    """支持通过 URL 参数 ?page=Anomaly%20Detection 直接打开指定页面。"""
    try:
        requested = st.query_params.get("page")
    except Exception:  # pragma: no cover - 兼容旧版本 Streamlit
        return 0
    if isinstance(requested, list):
        requested = requested[0] if requested else None
    if requested in PAGES:
        return PAGES.index(requested)
    return 0


def main() -> None:
    data = load_data()
    kpi = data["kpi"]

    page = st.sidebar.radio("页面导航", PAGES, index=_initial_page_index())

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**数据范围**\n\n{kpi.date_start} ~ {kpi.date_end}\n\n"
        f"**工单总数**：{kpi.total_tickets} 条\n\n"
        f"**未解决**：{kpi.unresolved} 条（{kpi.unresolved_rate:.1%}）\n\n"
        f"**平均满意度**：{kpi.avg_satisfaction}\n\n"
        f"**异常信号**：{len(data['anomalies'])} 条"
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "所有指标由 data/tickets.json 实时计算，未使用任何硬编码结果。"
        "异常信号为统计偏离提示，建议进一步确认。"
    )

    render = {
        "Overview": page_overview,
        "Trend Analysis": page_trend,
        "Category Analysis": page_category,
        "Priority Analysis": page_priority,
        "Resolution Analysis": page_resolution,
        "Satisfaction Analysis": page_satisfaction,
        "Anomaly Detection": page_anomaly,
        "Similar Tickets": page_similar,
    }[page]
    render(data)


if __name__ == "__main__":
    main()
