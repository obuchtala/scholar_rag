# Plan: Add Local Cache for Ingest Fetched Papers

## Goal

Persist the fetched+expanded paper list to disk after the S2 API phase so that
a failed or interrupted ingest can resume at the embed+upsert step without
repeating hours of API calls.

## Design

| Decision | Choice |
|---|---|
| Cache location | Configured via `CACHE_DIR` env var |
| Default `CACHE_DIR` | `cache/scholar-rag` (relative to project root) |
| Droplet | `/mnt/projects/scholar-rag/cache` (block storage, survives rebuilds) |
| Cache file | `$CACHE_DIR/papers.json` |
| Cache invalidation | `--no-cache` flag on `ingest` command |
| Cache format | JSON array of paper dicts (already plain dicts — no serialisation needed) |

## Files Changed

| File | Change |
|---|---|
| `src/scholar_rag/ingest.py` | Load cache if exists; save after expand; accept `cache_dir` param |
| `src/scholar_rag/cli.py` | Add `--no-cache` flag to `ingest` command |
| `.env.example` | Add `CACHE_DIR=` with comment |
| `.gitignore` | Add `cache/` |

## Steps

1. Update `.gitignore`
2. Update `.env.example`
3. Update `ingest.py` — `ingest()` function: load / save cache
4. Update `cli.py` — add `--no-cache` flag, pass to `ingest()`
