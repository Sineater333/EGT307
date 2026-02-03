# Step 1: Python image
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3. Prevent Python from writing .pyc files and enable unbuffered logging
# This helps with seeing logs in real-time in Docker/Kubernetes
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 4: Copy only the requirements first (to leverage Docker caching)
COPY requirements.txt .

# Step 5: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6. Copy the rest of the application code
COPY . .

# Step 7. Expose the port FastAPI runs on
EXPOSE 8000

# Step 8. Command to run the application
# We use host 0.0.0.0 so it's accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]