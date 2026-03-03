# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (ignoring warnings for now)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Fast API webserver
CMD ["uvicorn", "travel_brain.api.app:app", "--host", "0.0:0:0", "--port", "8000"]
