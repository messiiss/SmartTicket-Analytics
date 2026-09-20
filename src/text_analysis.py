"""重复 / 相似问题发现模块（TF-IDF + 余弦相似度）。

中文分词说明：项目不引入 jieba 等额外分词依赖，改用**字符 n-gram（1-2 gram）**
构造 TF-IDF 向量。对客服工单这种短文本 + 高频业务词（退款、扣款、快递）的场景，
字符级 n-gram 的表现足够稳定，且零额外依赖、可解释。

阈值说明：``DEFAULT_SIMILARITY_THRESHOLD`` 是基于当前数据集调出的经验参数，
不是绝对标准。Dashboard 侧提供滑块让使用者按自己的容忍度调整。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD: float = 0.30
MIN_DESCRIPTION_LENGTH: int = 4


def build_tfidf_matrix(descriptions: list[str]) -> tuple[Any, Any]:
    """构造 TF-IDF 矩阵。

    Returns:
        (vectorizer, matrix)；描述为空时返回 (None, None)。
    """
    cleaned = [text for text in descriptions if isinstance(text, str) and len(text.strip()) >= MIN_DESCRIPTION_LENGTH]
    if len(cleaned) < 2:
        logger.warning("有效描述不足 2 条，无法进行相似度分析")
        return None, None

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(cleaned)
    logger.debug("TF-IDF 矩阵：%s", matrix.shape)
    return vectorizer, matrix


def find_similar_pairs(
    df: pd.DataFrame,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_n: int | None = None,
) -> pd.DataFrame:
    """找出描述高度相似的工单对。

    Args:
        df: 清洗后的工单数据（需包含 ``description`` / ``ticket_id``）。
        threshold: 相似度阈值（0~1），越高越严格。
        top_n: 只返回相似度最高的前 N 对，None 表示全部。

    Returns:
        含 ``ticket_a`` / ``ticket_b`` / ``similarity`` / 描述 / 分类 / 优先级 / 满意度的表格。
    """
    columns = [
        "ticket_a",
        "ticket_b",
        "similarity",
        "desc_a",
        "desc_b",
        "category",
        "priority_a",
        "priority_b",
        "satisfaction_a",
        "satisfaction_b",
    ]
    if df is None or df.empty or "description" not in df.columns:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["description"] = working["description"].fillna("").astype(str).str.strip()
    working = working[working["description"].str.len() >= MIN_DESCRIPTION_LENGTH].reset_index(drop=True)
    if len(working) < 2:
        return pd.DataFrame(columns=columns)

    vectorizer, matrix = build_tfidf_matrix(working["description"].tolist())
    if vectorizer is None or matrix is None:
        return pd.DataFrame(columns=columns)

    similarity = cosine_similarity(matrix)

    rows: list[dict[str, Any]] = []
    for i in range(len(working)):
        for j in range(i + 1, len(working)):
            score = float(similarity[i, j])
            if score < threshold:
                continue
            rows.append(
                {
                    "ticket_a": working.loc[i, "ticket_id"],
                    "ticket_b": working.loc[j, "ticket_id"],
                    "similarity": round(score, 4),
                    "desc_a": working.loc[i, "description"],
                    "desc_b": working.loc[j, "description"],
                    "category": working.loc[i, "category"],
                    "priority_a": working.loc[i, "priority"],
                    "priority_b": working.loc[j, "priority"],
                    "satisfaction_a": working.loc[i, "satisfaction"],
                    "satisfaction_b": working.loc[j, "satisfaction"],
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        logger.info("阈值 %.2f 下未发现相似工单对", threshold)
        return result

    result = result.sort_values("similarity", ascending=False).reset_index(drop=True)
    if top_n:
        result = result.head(top_n)
    logger.info("阈值 %.2f 下发现 %d 组相似工单对", threshold, len(result))
    return result


def cluster_similar_tickets(df: pd.DataFrame, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> pd.DataFrame:
    """把相似工单聚成问题簇（连通分量法，不需要额外的聚类依赖）。

    Returns:
        含 ``cluster_id`` / ``工单数`` / ``ticket_ids`` / ``代表描述`` 的表格。
    """
    columns = ["cluster_id", "工单数", "ticket_ids", "代表描述", "涉及分类"]
    pairs = find_similar_pairs(df, threshold=threshold)
    if pairs.empty:
        return pd.DataFrame(columns=columns)

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for _, row in pairs.iterrows():
        union(str(row["ticket_a"]), str(row["ticket_b"]))

    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)

    desc_map = dict(zip(df["ticket_id"].astype(str), df["description"]))
    category_map = dict(zip(df["ticket_id"].astype(str), df["category"]))

    rows: list[dict[str, Any]] = []
    for index, members in enumerate(
        sorted(groups.values(), key=len, reverse=True), start=1
    ):
        if len(members) < 2:
            continue
        members = sorted(members)
        rows.append(
            {
                "cluster_id": f"C{index:02d}",
                "工单数": len(members),
                "ticket_ids": ", ".join(members),
                "代表描述": desc_map.get(members[0], ""),
                "涉及分类": " / ".join(
                    sorted({str(category_map.get(m, "未知")) for m in members})
                ),
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    logger.info("相似问题簇：%d 组", len(result))
    return result


def top_keywords(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """提取工单描述中的高频关键词（按 TF-IDF 权重聚合），供 Dashboard 展示。"""
    columns = ["keyword", "weight"]
    if df is None or df.empty or "description" not in df.columns:
        return pd.DataFrame(columns=columns)

    descriptions = df["description"].fillna("").astype(str).tolist()
    vectorizer, matrix = build_tfidf_matrix(descriptions)
    if vectorizer is None or matrix is None:
        return pd.DataFrame(columns=columns)

    weights = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(terms, weights), key=lambda item: item[1], reverse=True)

    filtered = [(term, weight) for term, weight in pairs if len(term) >= 2]
    # 去除被更长词覆盖的单字/重复组合，保持可读性
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for term, weight in filtered:
        if term in seen:
            continue
        seen.add(term)
        rows.append({"keyword": term, "weight": round(float(weight), 4)})
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows, columns=columns)
