FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

WORKDIR /app/app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
