FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app and static + model
COPY app.py ./
COPY static ./static
COPY winequality-red.onnx ./winequality-red.onnx

ENV PORT=8080
EXPOSE 8080
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1"]
