FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
# CPU-only torch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[boost,xai,api,dashboard,mlops]"
COPY api ./api
COPY dashboard ./dashboard
COPY configs ./configs
COPY experiments ./experiments
ENV ARGUS_MODEL_DIR=/models/model
EXPOSE 8000 8501 5000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
