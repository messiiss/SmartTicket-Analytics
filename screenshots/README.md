# 截图说明

本目录下的 Dashboard 截图均为**真实运行截图**：由 headless Chrome 访问本地正在运行的
Streamlit 服务（`http://localhost:8501`）逐页抓取，页面上的每一个数字都来自
`data/tickets.json` 的实时计算。

| 文件 | 对应页面 | 展示内容 |
| --- | --- | --- |
| `dashboard_overview.png` | Overview | 核心 KPI（总工单 50 / 已解决 42 / 未解决 8 / 平均处理 19.69h / 平均满意度 2.36 / 异常信号 8）+「主管关注」区 + 每日工单量与问题类型分布 + 自动主管摘要 |
| `trend_analysis.png` | Trend Analysis | 每日工单量折线、每日高优先级工单量、前后半段日均对比表 |
| `anomaly_analysis.png` | Anomaly Detection | 8 条异常信号（含判断依据与关联工单）、高优先级未解决工单表、异常类型 × 严重程度分布 |
| `category_analysis.png` | Category Analysis | 各类别数量、处理时长、满意度、风险类别交叉筛选 |
| `satisfaction_analysis.png` | Satisfaction Analysis | 满意度分布、类别 × 满意度热力图、各维度低满意度率、相关性表 |
| `similar_tickets.png` | Similar Tickets | 相似工单对、相似问题簇、高频关键词 |

## 待补录：`development_process.png`

**这张截图无法由程序代劳，需要开发者在自己机器上补录**（要求明确写了「不要伪造截图」）。
建议按下面的顺序操作，一台机器上就能截完：

```bash
# 1. 激活环境（截一张终端窗口，能看到虚拟环境与命令）
.venv\Scripts\activate

# 2. 跑单元测试（截一张 pytest 输出的终端截图，要求能看到 41 passed）
pytest -q

# 3. 跑完整分析流水线（截一张终端截图，能看到主管摘要输出）
python run_analysis.py

# 4. 启动 Dashboard（截一张终端截图，能看到 Local URL: http://localhost:8501）
streamlit run app.py
```

建议至少覆盖以下画面中的 2~3 个：

- **IDE 窗口**：能看到 `src/` 下的模块（data_loader / data_cleaner / anomaly_detection / text_analysis）与代码内容；
- **终端**：`pytest -q` 的 `41 passed`、`run_analysis.py` 的主管摘要输出；
- **AI 辅助工具窗口**：与 AI 讨论分析维度、异常阈值、README 结构的对话界面；
- **浏览器**：Streamlit 正在运行、地址栏显示 `localhost:8501`。

截好后保存为 `screenshots/development_process.png`，README 第 12 节已预留引用位置。
