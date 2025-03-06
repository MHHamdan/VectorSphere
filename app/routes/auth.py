from fastapi import APIRouter, HTTPException, Depends
from app.services.auth import hash_password, verify_password, create_access_token
from app.models.user import User

router = APIRouter()

fake_users_db = {}

@router.post("/register/")
async def register(user: User):
    """
    Register a new user.
    """
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    fake_users_db[user.username] = {
        "password": hash_password(user.password)
    }
    return {"message": "User registered successfully!"}

@router.post("/login/")
async def login(user: User):
    """
    Authenticate and log in a user.
    """
    db_user = fake_users_db.get(user.username)
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"access_token": token}

