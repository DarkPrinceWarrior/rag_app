\set ON_ERROR_STOP on

-- Disposable A/B clone only. Re-running against an already prepared database
-- must fail instead of accepting an extension or index with different bits.
CREATE EXTENSION pg_textsearch VERSION '1.3.1';

CREATE INDEX ix_chunks_bm25_ru_v1
ON chunks USING bm25 ((coalesce(text_ru, '') || E'\n' || coalesce(text_en, '')))
WITH (text_config='russian', k1=1.2, b=0.75);

CREATE INDEX ix_chunks_bm25_en_v1
ON chunks USING bm25 ((coalesce(text_en, '') || E'\n' || coalesce(text_ru, '')))
WITH (text_config='english', k1=1.2, b=0.75);
