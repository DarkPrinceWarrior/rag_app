FROM pgvector/pgvector:pg17@sha256:feb68f4f15446397d8cac7f4fe48fe4586de83160d1fc48b46283312d1a33966

ARG PG_TEXTSEARCH_VERSION=1.3.1
ARG PG_TEXTSEARCH_SHA256=dc9a823e16f59b24b8c7f17c07a497b728be6c043643b4d56a65c912fa254349
ARG PG_TEXTSEARCH_ASSET=pg-textsearch-v1.3.1-pg17-amd64.zip

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl unzip \
    && curl --fail --location --proto '=https' --tlsv1.2 \
       "https://github.com/timescale/pg_textsearch/releases/download/v${PG_TEXTSEARCH_VERSION}/${PG_TEXTSEARCH_ASSET}" \
       --output "/tmp/${PG_TEXTSEARCH_ASSET}" \
    && echo "${PG_TEXTSEARCH_SHA256}  /tmp/${PG_TEXTSEARCH_ASSET}" | sha256sum --check --strict \
    && unzip -q "/tmp/${PG_TEXTSEARCH_ASSET}" -d /tmp/pg_textsearch \
    && apt-get install --yes --no-install-recommends \
       "/tmp/pg_textsearch/pg-textsearch-postgresql-17_${PG_TEXTSEARCH_VERSION}-1_amd64.deb" \
    && rm -rf /var/lib/apt/lists/* /tmp/pg_textsearch "/tmp/${PG_TEXTSEARCH_ASSET}"

LABEL org.opencontainers.image.source="https://github.com/timescale/pg_textsearch"
LABEL org.opencontainers.image.version="1.3.1"
LABEL org.opencontainers.image.licenses="PostgreSQL"
