# Use RENCI python base image
FROM ghcr.io/translatorsri/renci-python-image:3.11.5

# Add image info
LABEL org.opencontainers.image.source https://github.com/Translator-CATRAX/shepherd

ENV PYTHONHASHSEED=0

WORKDIR /app

RUN chmod -R 777 .

COPY ./shepherd_utils ./shepherd_utils
COPY ./pyproject.toml .
RUN pip install .

COPY ./workers/xcrg_lookup/requirements.txt .
RUN pip install -r requirements.txt

USER nru

COPY ./workers/xcrg_lookup .

CMD ["python", "worker.py"]
