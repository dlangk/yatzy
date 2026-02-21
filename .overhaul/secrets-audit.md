# Secrets & Sensitive Data Audit

**Date:** 2026-02-21
**Scanned by:** Claude Code (Phase 1 of OVERHAUL_PLAN.md)

## Scan Methodology

Scanned all source files (`*.rs`, `*.ts`, `*.js`, `*.py`, `*.json`, `*.yml`, `*.sh`, `*.html`, `*.toml`) excluding `target/`, `node_modules/`, `.git/`, `data/`, `outputs/`, lock files, and binary artifacts.

### Patterns Tested

| Pattern | Description | Result |
|---------|-------------|--------|
| `API_KEY=`, `SECRET_KEY=`, `AUTH_TOKEN=` | Hardcoded API keys | Clean |
| `Bearer <token>` | Embedded bearer tokens | Clean |
| `-----BEGIN PRIVATE KEY` | PEM private keys | Clean |
| `password=`, `passwd=`, `pwd=` | Hardcoded passwords | Clean |
| `postgres://`, `mysql://`, `redis://`, `mongodb://` | Connection strings with credentials | Clean |
| `AKIA[0-9A-Z]{16}` | AWS access key IDs | Clean |
| `aws_access_key_id`, `aws_secret_access_key` | AWS credential assignments | Clean |
| `ghp_*`, `glpat-*` | GitHub/GitLab tokens | Clean |
| `hooks.slack.com/services/` | Webhook URLs | Clean |
| Base64 blobs (>64 chars in assignments) | Encoded secrets | Clean |
| `http(s)://user:pass@host` | URLs with embedded credentials | Clean |
| Email addresses in source | PII exposure | Clean |

## Findings

### No Secrets Found

The codebase contains **zero hardcoded secrets**. No API keys, tokens, passwords, private keys, connection strings, or cloud credentials were detected in any source file.

### Environment Variable Handling

| Component | Env Var | Usage | Status |
|-----------|---------|-------|--------|
| Docker compose | `API_BASE_URL` | Frontend API target | Hardcoded to `http://localhost:9000` in compose file |
| Docker compose | `API_PORT` | Backend port | Hardcoded to `9000` in compose file |
| Solver Dockerfile | `YATZY_BASE_PATH` | Data directory root | Hardcoded to `/app` (appropriate for container) |
| Frontend | `__API_BASE_URL__` | Runtime config injection | Properly injected via `entrypoint.sh` from env |
| Frontend | `VITE_API_BASE_URL` | Build-time config | Vite env var (clean) |
| Solver | `YATZY_BASE_PATH` | Data directory root | Read from env at runtime (clean) |
| Solver | `RAYON_NUM_THREADS` | Parallelism | Read from env at runtime (clean) |

### `.gitignore` Coverage

`.env` is properly listed in `.gitignore` (line 41). No `.env` files exist in the repository.

### Docker Security

- `docker-compose.yml`: Uses `API_BASE_URL` env var for frontend (hardcoded `http://localhost:9000` is a development default, acceptable)
- `solver/Dockerfile`: Multi-stage build, no secrets baked in
- `frontend/Dockerfile`: Multi-stage build, `entrypoint.sh` injects config from env at container start
- No `.env` files are COPY'd into images

### CI/CD

No `.github/` directory exists — no CI/CD pipeline to audit.

## Recommendations

### Low Priority (Informational Only)

1. **Docker compose hardcoded defaults** — `API_BASE_URL=http://localhost:9000` and `API_PORT=9000` are hardcoded in `docker-compose.yml`. These are development defaults and not secrets, but could use `${VAR:-default}` syntax for cleaner override support.

2. **No `.env.example`** — Creating one would document the available environment variables for new developers.

3. **No pre-commit secret scanning** — Consider adding `detect-secrets` or `gitleaks` as a pre-commit hook to prevent accidental secret commits in the future.

## Remediation Applied

### 1. Created `.env.example`

See `/.env.example` for documented environment variables with placeholder values.

### 2. Pre-commit Hook Configuration

A `gitleaks.toml` configuration is provided at `/.gitleaks.toml` for use with the [gitleaks](https://github.com/gitleaks/gitleaks) secret scanner.

To install as a pre-commit hook:
```bash
# Install gitleaks
brew install gitleaks

# Run manually
gitleaks detect --source . --verbose

# Or add to .pre-commit-config.yaml:
# - repo: https://github.com/gitleaks/gitleaks
#   rev: v8.18.0
#   hooks:
#     - id: gitleaks
```

## Verdict

**CLEAN** — No blocking issues. The codebase handles configuration via environment variables appropriately. Proceed to Phase 2.
