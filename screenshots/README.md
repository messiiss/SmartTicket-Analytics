# 截图说明

本目录下的 Dashboard 截图均为**真实运行截图**：由 headless Chrome 访问正在运行的
Streamlit 服务逐页抓取，页面上的每一个数字都来自 `data/tickets.json` 的实时计算。

## 页面整图

| 文件 | 对应页面 | 展示内容 |
| --- | --- | --- |
| `dashboard_overview.png` | Overview | 核心 KPI（总工单 50 / 已解决 42 / 未解决 8 / 平均处理 19.69h / 平均满意度 2.36 / 异常信号 8）+「主管关注」区 + 每日工单量与问题类型分布 + 自动主管摘要 |
| `trend_analysis.png` | Trend Analysis | 每日工单量折线、每日高优先级工单量、前后半段日均对比表 |
| `anomaly_analysis.png` | Anomaly Detection | 8 条异常信号（含判断依据与关联工单）、高优先级未解决工单表、异常类型 × 严重程度分布 |
| `category_analysis.png` | Category Analysis | 各类别数量、处理时长、满意度、风险类别交叉筛选 |
| `satisfaction_analysis.png` | Satisfaction Analysis | 满意度分布、类别 × 满意度热力图、各维度低满意度率、相关性表 |
| `similar_tickets.png` | Similar Tickets | 相似工单对、相似问题簇、高频关键词 |

## 细节放大图

整页截图较宽（2240px），正文文字在 README 里会偏小。以下三张是对**最关键证据区**
按 1.6 倍设备像素比单独裁切的放大图，嵌入 README 第 12 节，用于直接阅读文字内容：

| 文件 | 裁切区域 | 为什么需要放大 |
| --- | --- | --- |
| `detail_manager_focus.png` | Overview 的「主管关注」区前两条信号 | 需要看清每条信号的**判断依据**与关联工单编号 |
| `detail_anomaly_evidence.png` | Anomaly 页第 1 条信号卡（【高】高优先级未解决） | 需要看清 `priority=高 且 is_resolved=false`、`7/8`、`120 小时` 等判定细节 |
| `detail_similar_clusters.png` | Similar 页「相似问题簇」表 | 需要看清 C01 / C02 两个簇的工单编号与代表描述 |

## 线上部署与开发过程

| 文件 | 说明 |
| --- | --- |
| `deployed_cloud.png` | **云服务器线上实例**（<http://134.185.89.68:8501>）的真实截图，用于证明 Docker 部署后数据、中文与图表全部正常 |
| `development_requirements.png` | 开发过程①：拿到题目后拆解 15 项核心要求，映射为「数据层 / 指标层 / 异常层 / 展示层」四个模块 |
| `development_ai_chat.png` | 开发过程②：AI 协作记录（技术选型讨论、模块拆分、README 结构整理） |

## 可继续补充的画面（可选）

如果希望「开发过程」这一项覆盖得更完整，可以再补 1~2 张本机画面：

```bash
.venv\Scripts\activate     # ① 终端：虚拟环境
pytest -q                  # ② 终端：41 passed
python run_analysis.py     # ③ 终端：主管摘要输出
streamlit run app.py       # ④ 终端：Local URL: http://localhost:8501
docker compose up -d       # ⑤ 终端：容器启动
```

优先建议补：**IDE 窗口**（打开 `src/anomaly_detection.py`，能看到异常检测的判定逻辑）与
**终端 pytest 输出**（能看到 `41 passed`）——这两张最能说明「代码是自己写的、测试是自己跑的」。
