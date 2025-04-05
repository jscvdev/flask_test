FROM python:3.10-slim

# Install system dependencies like libGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory
WORKDIR /app

# Copy the requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port and run the app
EXPOSE 5000
CMD ["gunicorn", "app:app"]
