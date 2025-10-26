FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy metrics exporter
COPY metrics_exporter.py .

# Expose metrics port
EXPOSE 9091

# Start metrics server
CMD ["python3", "-c", "from metrics_exporter import start_metrics_exporter; start_metrics_exporter(); import time; time.sleep(1000000)"]

