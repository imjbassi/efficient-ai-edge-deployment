# Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set system environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set the working directory in the container
WORKDIR /app

# Install minimal system dependencies required for image decoding and common utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for models and source code
RUN mkdir -p models src

# Copy the source code and model files into the container
COPY src/ ./src/

# Verify if models are available, or create mock folder if mounting at runtime
# Copy pre-trained models if they exist in models folder
COPY models/ ./models/

# Expose the API port
EXPOSE 8000

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
