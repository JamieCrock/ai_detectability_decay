FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake in the demo model so the image runs out of the box. To serve a model
# trained on real data instead, mount it and point DETECTABILITY_MODEL at it:
#   docker run -p 8000:8000 -v $PWD/models:/models \
#     -e DETECTABILITY_MODEL=/models/random_forest.joblib ai-detectability
RUN detectability demo --out /app/models

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["detectability", "serve", "--host", "0.0.0.0", "--port", "8000"]
