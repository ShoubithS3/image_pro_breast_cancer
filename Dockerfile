FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.23
COPY pyproject.toml uv.lock README.md ./
COPY mammo_lens ./mammo_lens
RUN uv sync --locked --extra train --no-dev
COPY frontend ./frontend
COPY scripts ./scripts
RUN useradd --create-home researcher && mkdir -p /app/data /app/artifacts /app/state && chown -R researcher:researcher /app
USER researcher
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "mammo_lens.api:app", "--host", "0.0.0.0", "--port", "8000"]
