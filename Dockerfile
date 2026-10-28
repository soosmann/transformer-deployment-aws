FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY api_main.py model_serve_streamlit.py ./

EXPOSE 8000 8501

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
