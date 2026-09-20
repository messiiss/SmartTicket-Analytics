"""命令行入口：跑完整分析流水线并输出 Markdown 报告 + 控制台摘要。

用法::

    python run_analysis.py                 # 使用 data/tickets.json
    python run_analysis.py --threshold 0.35 --out outputs/report.md

该脚本与 Streamlit Dashboard 共用同一套 src 模块，保证「报告」与「页面」数字一致。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import detect_all_anomalies, signals_dataframe  # noqa: E402
from src.data_cleaner import clean_tickets, summarize_data_quality  # noqa: E402
from src.data_loader import load_tickets  # noqa: E402
from src.metrics import (  # noqa: E402
    correlation_analysis,
    group_summary,
    high_risk_categories,
    long_tail_tickets,
    overall_kpis,
    satisfaction_distribution,
)
from src.report_generator import build_markdown_report, generate_manager_summary, save_report  # noqa: E402
from src.text_analysis import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    cluster_similar_tickets,
    find_similar_pairs,
    top_keywords,
)
from src.trend_analysis import daily_counts, period_daily_rates, suggest_split_date, trend_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartTicket Analytics 分析流水线")
    parser.add_argument("--data", default=None, help="tickets.json 路径，默认 data/tickets.json")
    parser.add_argument("--out", default="outputs/report.md", help="报告输出路径")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="相似工单判定阈值（经验参数，默认 0.30）",
    )
    parser.add_argument("--verbose", action="store_true", help="输出 DEBUG 日志")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("run_analysis")

    load_result = load_tickets(args.data)
    clean_result = clean_tickets(load_result.dataframe)
    df = clean_result.dataframe

    logger.info("已加载 %d 条工单，清洗后 %d 条", load_result.row_count, len(df))
    if load_result.missing_fields:
        logger.warning("数据缺失必需字段：%s", load_result.missing_fields)

    kpi = overall_kpis(df)
    trend = trend_summary(df)
    category_summary = group_summary(df, "category")
    priority_summary = group_summary(df, "priority")
    channel_summary = group_summary(df, "channel")
    satisfaction_table = satisfaction_distribution(df)
    long_tail = long_tail_tickets(df)
    similar_pairs = find_similar_pairs(df, threshold=args.threshold)
    clusters = cluster_similar_tickets(df, threshold=args.threshold)
    anomalies = detect_all_anomalies(df, cluster_table=clusters, split_date=suggest_split_date(df))
    quality = summarize_data_quality(clean_result)

    summary_text = generate_manager_summary(
        kpi=kpi,
        trend=trend,
        category_summary=category_summary,
        anomaly_signals=anomalies,
        repeat_cluster_count=len(clusters),
    )

    tables: dict[str, pd.DataFrame] = {
        "问题类型分布": category_summary,
        "优先级分布": priority_summary,
        "渠道分布": channel_summary,
        "满意度分布": satisfaction_table,
        "每日工单量": daily_counts(df),
        "前后半段日均对比": period_daily_rates(df, suggest_split_date(df) or ""),
        "长时间处理工单（IQR 上界）": long_tail,
        "需重点关注的类别（客观指标交叉）": high_risk_categories(df),
        "维度相关性（Spearman）": correlation_analysis(df),
        "相似工单对": similar_pairs.head(20),
        "相似问题簇": clusters,
        "高频关键词": top_keywords(df),
        "异常信号列表": signals_dataframe(anomalies),
    }

    report = build_markdown_report(
        kpi=kpi,
        trend=trend,
        summary_text=summary_text,
        tables=tables,
        anomaly_signals=anomalies,
        quality_summary=quality,
    )
    out_path = save_report(report, PROJECT_ROOT / args.out)

    print("\n" + "=" * 78)
    print(summary_text)
    print("=" * 78)
    print(f"\n完整报告：{out_path}")
    print(f"相似工单对（阈值 {args.threshold}）：{len(similar_pairs)} 组；相似问题簇：{len(clusters)} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
