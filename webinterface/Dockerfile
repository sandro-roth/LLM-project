FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY app /app/app

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
