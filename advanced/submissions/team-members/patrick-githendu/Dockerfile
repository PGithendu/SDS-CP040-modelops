FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY patrick.py .
COPY model.pkl .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["uvicorn", "patrick:app", "--host", "0.0.0.0", "--port", "10000"]