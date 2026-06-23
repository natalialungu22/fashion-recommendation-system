FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app/app.py"]