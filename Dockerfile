# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app will run on
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app/main.py
ENV FLASK_ENV=production

# Run the Flask app
CMD ["python", "app/main.py"]
