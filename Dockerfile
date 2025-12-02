# Use a small official Python image
FROM python:3.10-slim

# Create a working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run your Flask app
CMD ["python", "app.py"]
