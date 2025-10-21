FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY patrick.py .
COPY model.pkl .
COPY index.html .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000
RUN pip install uvicorn
CMD ["uvicorn", "patrick:app", "--host", "0.0.0.0", "--port", "10000"]