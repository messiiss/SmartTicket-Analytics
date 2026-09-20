# SmartTicket Analytics —— 生产镜像
#
# 基础镜像说明：python:3.11-slim 同时提供 amd64 与 arm64 两个架构的 manifest，
# 因此本地（x86_64）与云服务器（Oracle Cloud aarch64）都能直接构建，无需 --platform。
FROM python:3.11-slim

LABEL org.opencontainers.image.title="SmartTicket Analytics" \
      org.opencontainers.image.description="客服工单趋势与异常分析平台" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 先装依赖再拷代码：依赖层可被 Docker 缓存，改代码时不必重装 pandas/sklearn
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest

# 只拷运行所需内容（.dockerignore 已排除 .git / .venv / 缓存 / 截图等）
COPY app.py run_analysis.py ./
COPY src/ ./src/
COPY data/ ./data/

# 以非 root 用户运行，降低容器逃逸风险
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# 镜像内没有 curl，用 Python 标准库做健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
