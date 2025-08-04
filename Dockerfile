FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY predict_api.py /app/predict_api.py

RUN pip install --no-cache-dir fastapi uvicorn mlflow pandas

EXPOSE 8000

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]