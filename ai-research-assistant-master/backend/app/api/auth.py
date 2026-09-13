"""Authentication endpoints for local JWT login."""

from fastapi import APIRouter, HTTPException, status, Depends

from app.core.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
)
from app.models.schemas import (
    AuthTokenResponse,
    AuthUser,
    UserLoginRequest,
    UserRegisterRequest,
)

router = APIRouter()


@router.post("/auth/register", response_model=AuthTokenResponse)
async def register(request: UserRegisterRequest):
    user = create_user(request.name, request.email, request.password)
    token = create_access_token(user)
    return AuthTokenResponse(access_token=token, user=AuthUser(**user))


@router.post("/auth/login", response_model=AuthTokenResponse)
async def login(request: UserLoginRequest):
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    token = create_access_token(user)
    return AuthTokenResponse(access_token=token, user=AuthUser(**user))


@router.get("/auth/me", response_model=AuthUser)
async def me(current_user: dict = Depends(get_current_user)):
    return AuthUser(**current_user)