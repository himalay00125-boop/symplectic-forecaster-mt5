FROM python:3.11-slim

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install mt5linux and rpyc for bridging to Windows MT5
RUN pip install --no-cache-dir mt5linux rpyc

# Copy script and dashboard assets
COPY symplectic_forecaster.py .
COPY dashboard.html .

# Set container environments
ENV IS_DOCKER=true
ENV MT5_HOST=host.docker.internal

# Expose live dashboard port
EXPOSE 8080

# Set container entrypoint
ENTRYPOINT ["python", "symplectic_forecaster.py"]
CMD ["--symbol", "EURUSD", "--timeframe", "H1"]
