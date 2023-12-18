# Use the official Python image as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Flask will run on
EXPOSE 5000

# Set environment variables for cloud credentials
ENV CLOUD_CREDENTIALS_FILE=/app/dazzling-tensor-405719-b0b850808aff

# Copy the data folder into the image
COPY data /app/data

# Define the command to run your application
CMD ["python", "your_flask_app.py"]
