# Use official Python image as base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy only the backend folder
COPY backend /app

# Set working directory to backend
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data inside Docker container
RUN python -m nltk.downloader punkt stopwords

# Expose the port Flask runs on
EXPOSE 5000

# Run the application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
