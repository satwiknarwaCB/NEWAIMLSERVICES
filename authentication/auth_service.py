import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import hashlib
import jwt
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Authentication Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("AUTH_DATABASE", "auth_db")

# Connect to MongoDB
client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]
users_collection = db.users

# Pydantic models
class UserRegister(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None
    role: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserRoleUpdate(BaseModel):
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserInfo(BaseModel):
    id: str
    email: str
    display_name: Optional[str] = None
    role: Optional[str] = None

# Helper functions
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(authorization: str = Header(None)):
    """Extract user from authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if not user_email:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        user = users_collection.find_one({"email": user_email})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
            
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

# Routes
@app.post("/auth/register", response_model=Token)
async def register(user: UserRegister):
    """Register a new user"""
    
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password and store user
    hashed_password = hash_password(user.password)
    user_data = {
        "email": user.email,
        "password": hashed_password,
        "display_name": user.display_name or user.email.split("@")[0],
        "role": user.role or "public",
        "created_at": datetime.now()
    }
    
    # Insert user into MongoDB
    result = users_collection.insert_one(user_data)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": str(result.inserted_id),
            "email": user.email,
            "display_name": user_data["display_name"],
            "role": user_data["role"]
        }
    )

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """Login user"""
    
    # Check if user exists
    stored_user = users_collection.find_one({"email": user.email})
    if not stored_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    hashed_password = hash_password(user.password)
    
    if hashed_password != stored_user["password"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": str(stored_user["_id"]),
            "email": user.email,
            "display_name": stored_user.get("display_name", user.email.split("@")[0]),
            "role": stored_user.get("role", "public")
        }
    )

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user"""
    return {"message": "Successfully logged out"}

@app.post("/auth/refresh", response_model=Token)
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """Refresh access token"""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user["email"]}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": str(current_user["_id"]),
            "email": current_user["email"],
            "display_name": current_user.get("display_name", current_user["email"].split("@")[0]),
            "role": current_user.get("role", "public")
        }
    )

@app.get("/auth/user", response_model=UserInfo)
async def get_user_info(current_user: dict = Depends(get_current_user)):
    """Get user information"""
    return UserInfo(
        id=str(current_user["_id"]),
        email=current_user["email"],
        display_name=current_user.get("display_name", current_user["email"].split("@")[0]),
        role=current_user.get("role", "public")
    )

@app.put("/auth/user/role", response_model=UserInfo)
async def update_user_role(role_update: UserRoleUpdate, current_user: dict = Depends(get_current_user)):
    """Update user role"""
    # Update user role in database
    users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"role": role_update.role}}
    )
    
    # Return updated user info
    updated_user = users_collection.find_one({"_id": current_user["_id"]})
    
    return UserInfo(
        id=str(updated_user["_id"]),
        email=updated_user["email"],
        display_name=updated_user.get("display_name", updated_user["email"].split("@")[0]),
        role=updated_user.get("role", "public")
    )

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Authentication Service Running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "authentication"}

if __name__ == "__main__":
    print("🚀 Starting Authentication Service...")
    print("📡 Server will be available at: http://localhost:8080")
    print("📚 API Documentation at: http://localhost:8080/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "auth_service:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )