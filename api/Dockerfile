# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY api/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code and model artifacts
COPY api/ /app/api/
COPY scripts/train.py /app/scripts/train.py
COPY data/ /app/data/
COPY models/ /app/models/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]