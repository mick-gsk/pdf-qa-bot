FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY pdfqa ./pdfqa
RUN pip install --no-cache-dir ".[anthropic,openai]"
ENV DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "pdfqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
