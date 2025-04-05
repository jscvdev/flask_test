# Start with a lightweight Python image
FROM python:3.10-slim

# Install system dependencies, including libGL for OpenCV and other required packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    # Check if libGL is installed
    && ldconfig -p | grep libGL
    
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the Flask app's port (5000 by default)
EXPOSE 5000

# Run the app using gunicorn (Flask's production WSGI server)
CMD ["gunicorn", "app:app"]
