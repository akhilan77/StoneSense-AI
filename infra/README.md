# StoneSense-AI Infrastructure & Deployment

The `infra/` directory contains containerization templates, reverse proxy configurations, and orchestration files for production and local development.

---

## Directory Layout

```text
infra/
├── docker/
│   ├── Dockerfile.backend    # Multi-stage production container for FastAPI backend
│   └── Dockerfile.frontend   # Node.js build stage + Nginx web server for React frontend
└── nginx/
    └── nginx.conf            # Reverse proxy configuration routing /api/ and WebSockets
```

---

## Docker Compose Orchestration

The root [`docker-compose.yml`](../docker-compose.yml) orchestrates 3 services:
1. **`postgres`**: PostgreSQL 16 Alpine container with persistent named volume `postgres_data`.
2. **`backend`**: FastAPI application connected to Postgres with health check dependencies.
3. **`frontend`**: Nginx web server serving compiled React static assets on port 80 and proxying API traffic to port 8000.

### Starting Services via Docker Compose
```bash
# 1. Populate secrets in .env
cp .env.example .env

# 2. Build and start containers in background
docker compose up --build -d

# 3. Apply database migrations to head
docker compose exec backend alembic upgrade head

# 4. Seed initial demonstration users and hospitals
docker compose exec backend python -m backend.app.db.seed
```

---

## Security & Environment Variables

| Variable | Description | Default (Development) |
| :--- | :--- | :--- |
| `STONESENSE_ENV` | Environment identifier | `development` |
| `STONESENSE_DATABASE_URL` | Database connection string | `sqlite:///stonesense.db` |
| `STONESENSE_JWT_SECRET` | Secret key for HMAC SHA-256 JWT signing | *Must be $\ge 32$ chars* |
| `CORS_ORIGINS` | Allowed cross-origin frontend domains | `http://localhost:5173,http://localhost:80` |
| `GATE_MIN_ACCURACY` | Minimum accuracy threshold for deployment gate | `0.95` |
| `GATE_MIN_F1` | Minimum Macro F1 threshold for deployment gate | `0.94` |
| `GATE_MIN_RECALL_STONE` | Minimum Stone sensitivity threshold | `0.96` |
| `GATE_MIN_RECALL_TUMOR` | Minimum Tumor sensitivity threshold | `0.98` |
