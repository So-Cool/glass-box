# The original Glass-Box natural language understanding model implemented
# with Rasa NLU 0.12.3 (Python <= 3.6) and served over HTTP.
FROM python:3.6-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "numpy==1.13.3" "scipy==0.19.1" "scikit-learn==0.19.1" \
      "thinc==6.10.3" "spacy==2.0.8" "sklearn_crfsuite==0.3.6" \
 && pip install --no-cache-dir \
      "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz" \
 && python -m spacy link en_core_web_sm en

ARG RASA_REF=e7c34eb2ebb3a33a047b56a33fdc8f8eedee95c2
RUN pip install --no-cache-dir "git+https://github.com/RasaHQ/rasa.git@${RASA_REF}"

COPY projects /app/projects
WORKDIR /app
EXPOSE 5005
CMD ["python", "-m", "rasa_nlu.server", "--path", "/app/projects", "--port", "5005"]
