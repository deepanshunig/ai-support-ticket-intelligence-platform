FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-lock.txt /app/requirements-lock.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r /app/requirements-lock.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

COPY --chown=app:app pyproject.toml README.md /app/
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app ui/ /app/ui/
COPY --chown=app:app configs/ /app/configs/

USER app

EXPOSE 8000 8501

CMD ["python", "-m", "ticket_intelligence.api.main"]
