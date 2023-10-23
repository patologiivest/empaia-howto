FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN python3 -m pip install -r Requirements.txt
CMD ["python3", "glue.py"]
