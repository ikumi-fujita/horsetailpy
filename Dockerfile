FROM python:3.12.8-slim-bookworm

COPY ./pyproject.toml  ./
RUN apt-get update && apt-get install -y sudo curl bash git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

RUN mkdir -p /work
WORKDIR /work
