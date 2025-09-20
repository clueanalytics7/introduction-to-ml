FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config
COPY .streamlit/ .streamlit/
COPY apps/app_classifier.py .

# Set environment variables to prevent permission errors
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Create a directory that Streamlit can write to
RUN mkdir -p /tmp/streamlit && chmod 777 /tmp/streamlit
ENV STREAMLIT_CACHE_DIR=/tmp/streamlit

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Run Streamlit with specific settings to avoid permission issues
ENTRYPOINT ["streamlit", "run", "app_classifier.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--global.developmentMode=false", \
    "--browser.serverAddress=0.0.0.0", \
    "--logger.level=error"]
