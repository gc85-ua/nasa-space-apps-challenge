FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY ./main.py /app/main.py
COPY ./frontend /app/frontend
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml
COPY .python-version /app/.python-version
RUN uv remove scikit-learn
RUN uv remove numpy
RUN uv remove matplotlib
RUN uv remove pandas
RUN uv sync
EXPOSE 80

CMD ["uv", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]