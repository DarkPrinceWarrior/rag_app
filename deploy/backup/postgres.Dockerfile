ARG PGVECTOR_IMAGE=pgvector/pgvector:0.8.5-pg17-bookworm@sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0
FROM ${PGVECTOR_IMAGE}

RUN apt-get update \
    && apt-get install --no-install-recommends -y pgbackrest \
    && rm -rf /var/lib/apt/lists/*
