# One image for both services; docker-compose.yml picks the command.
# Build: docker build -t fraudlens .   Run the API: docker run -p 8000:8000 fraudlens
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# xgboost's wheel needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml config.yaml ./
COPY src ./src
COPY app ./app
COPY models ./models
COPY reports ./reports
# editable, so fraud.config finds config.yaml and models/ relative to /app
RUN pip install --no-deps -e .

EXPOSE 8000 8501
CMD ["uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
