# Use an official Python runtime as a parent image
# We choose a slim-buster image for smaller size and better security
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages (e.g., for sentence-transformers)
# This includes build-essential for compiling some packages if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed Python packages specified in requirements.txt
# --no-cache-dir reduces image size
# --upgrade pip ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
# The /app directory in the container is now populated with your source code
COPY . .

# Expose the port that FastAPI will run on inside the container
# DigitalOcean App Platform and Render will map an external port to this internal port
EXPOSE 8000

# Command to run the application using Uvicorn
# This is the default command that will be executed when the container starts
# The --host 0.0.0.0 is crucial for the app to be accessible from outside the container
# We use an environment variable $PORT, which Render/DigitalOcean injects,
# but inside the Dockerfile, we hardcode 8000 as the EXPOSEd port.
# The Procfile will override this for DigitalOcean App Platform, but it's good to have a default.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
