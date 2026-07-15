# =============================================================================
# Base Image
# =============================================================================

# Lightweight official Python image
FROM python:3.12-slim

# =============================================================================
# Environment Variables
# =============================================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# =============================================================================
# Working Directory
# =============================================================================

WORKDIR /app

# =============================================================================
# Install System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Install Python Dependencies
# =============================================================================

# Copy only the requirements first to leverage Docker layer caching
COPY requirements-prod.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# =============================================================================
# Copy Application Source Code
# =============================================================================

# Copies the complete project into the container.
# The docker/ directory (including gunicorn.conf.py) is copied as well.
COPY . .

# =============================================================================
# Create a Non-Root User
# =============================================================================

RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

# =============================================================================
# Expose Application Port
# =============================================================================

EXPOSE 8000

# =============================================================================
# Start FastAPI Application using Gunicorn
# =============================================================================

CMD ["gunicorn", "-c", "docker/api/gunicorn.conf.py", "api:app"]