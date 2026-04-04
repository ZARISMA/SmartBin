FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libusb-1.0-0 libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    opencv-python-headless psycopg2-binary fastapi "uvicorn[standard]" jinja2

COPY . .

EXPOSE 8000

CMD ["python", "-m", "smartwaste.web"]
