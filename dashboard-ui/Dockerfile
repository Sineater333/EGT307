# Step 1: Python image
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the requirements first (to leverage Docker caching)
COPY requirements.txt .
# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard script
COPY dashboard.py .

# Streamlit uses port 8501 by default
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
