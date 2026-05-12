"""API routers aggregated under ``/api/v1``."""

from fastapi import APIRouter

from nextext.api.routes import health, jobs

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(jobs.router)

__all__ = ["router"]
