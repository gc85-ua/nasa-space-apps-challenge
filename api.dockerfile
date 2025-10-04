FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /api
COPY ./ml /api/ml
COPY ./backend/app/ /api/app/
COPY uv.lock /api/uv.lock
COPY pyproject.toml /api/pyproject.toml
COPY .python-version /api/.python-version
RUN uv sync
WORKDIR /api/app
EXPOSE 80
CMD [ "uv", "run", "fastapi", "run", "--port", "80" ]