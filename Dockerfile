FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY apps/app_classifier.py .

# Expose port and run Streamlit
EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health
ENTRYPOINT ["streamlit", "run", "app_classifier.py", "--server.port=7860", "--server.address=0.0.0.0"]
