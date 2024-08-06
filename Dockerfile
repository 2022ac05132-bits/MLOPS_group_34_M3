# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app/app.py"]
