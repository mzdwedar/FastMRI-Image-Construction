FROM python:3.12-slim
COPY sink.py /app/sink.py
USER nobody
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["python", "-u", "/app/sink.py"]
RUN apt-get update && apt-get install -y curl
