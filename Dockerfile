FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (cache layer if code changes)
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Now copy rest of the code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
