"""JWT authentication and simple role-based access control for the demo API."""
import os
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
USERS = {"researcher": {"password": "researcher-demo", "role": "researcher"}, "operator": {"password": "operator-demo", "role": "operator"}, "admin": {"password": "admin-demo", "role": "admin"}}

def issue_token(username: str, role: str):
    return jwt.encode({"sub": username, "role": role, "exp": datetime.now(timezone.utc) + timedelta(hours=8)}, SECRET, algorithm="HS256")

def current_user(token=Depends(oauth2)):
    if not AUTH_ENABLED: return {"sub": "local-demo", "role": "admin"}
    if not token: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    try: return jwt.decode(token, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token")

def require_role(*roles):
    def dependency(user=Depends(current_user)):
        if user.get("role") not in roles: raise HTTPException(status_code=403, detail="insufficient role")
        return user
    return dependency
