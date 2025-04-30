# Use an official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies from correct path
RUN pip install --no-cache-dir -r app/requirements.txt

# Expose the port your app will run on
EXPOSE 8080

# Run the Flask app
CMD ["python", "app/main.py"]
