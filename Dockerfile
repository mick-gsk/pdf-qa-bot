FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY pdfqa ./pdfqa
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[anthropic,openai]"

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
USER app
WORKDIR /home/app
ENV DATA_DIR=/data PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "pdfqa.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
