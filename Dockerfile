```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs are immediately displayed
ENV PYTHONUNBUFFERED=1

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Copy dataset
COPY data ./data

# Copy pre-trained models
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
