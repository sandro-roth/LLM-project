# Use lightweight, stable python image FROM python:3.11-slim
FROM python:3.11-slim

# Environment configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy the entire app folder into the container
COPY app /app/app

# Expose the FastAPI port
EXPOSE 8100

# Run the FastAPI server using uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]
