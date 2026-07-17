from fastapi import HTTPException, Request

from .config import settings


async def require_auth(request: Request) -> None:
    if not settings.api_key:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {settings.api_key}":
        raise HTTPException(401, "invalid or missing bearer token")
