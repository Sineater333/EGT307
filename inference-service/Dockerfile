# Step 1: Python image
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the requirements first (to leverage Docker caching)
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the code, the models folder, and the data folder
COPY main.py .
COPY models/ ./models/

# Step 6: Create the logs directory so the CSV has a place to live
RUN mkdir -p logs

# Step 7: Expose the port FastAPI runs on
EXPOSE 8000

# Step 8: Command to run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]