# fastapi.py - Cognitbotz AI Legal Platform - FastAPI Backend

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import tempfile
import uuid
import sqlite3
import glob
import jwt
from dotenv import load_dotenv
from io import BytesIO

# PDF and DOCX generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Load environment
load_dotenv()

# Import our modules
from vectors import EmbeddingsManager
from chatbot import ChatbotManager
from legal_drafting import LegalDraftingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx', 'pptx']
MAX_INTERACTIONS_PER_SESSION = 25
MAX_TOKENS_PER_SESSION = 75000
PERSISTENT_COLLECTION_NAME = "Legal_documents"
DOCUMENTS_FOLDER = Path(__file__).parent / "documents"

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Initialize FastAPI
app = FastAPI(
    title="Cognitbotz AI Legal Platform API",
    description="Backend API for legal document analysis and drafting",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Hardcoded users (same as rag.py)
USERS_DB = {
    "admin": {
        "password": "Admin@123",
        "role": "Administrator",
        "full_name": "System Administrator"
    },
    "finance": {
        "password": "Finance@123",
        "role": "Finance User",
        "full_name": "Finance Department"
    },
    "requester": {
        "password": "Request@123",
        "role": "Purchase Requester",
        "full_name": "Purchase Requester"
    }
}

# In-memory session storage (use Redis in production)
active_sessions = {}

# ==================== PYDANTIC MODELS ====================

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, str]

class ChatRequest(BaseModel):
    message: str
    chatbot_mode: str = "Document Only"
    chat_session_id: Optional[str] = None
    layman_mode: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int
    is_flagged: bool
    flag_reason: Optional[str]
    response_type: str
    chat_session_id: str

class DraftingRequest(BaseModel):
    doc_type: str
    requirements: str
    style: str = "Formal Legal"
    length: str = "Standard"
    clauses: Optional[List[str]] = None
    special_provisions: Optional[str] = None

class DraftingResponse(BaseModel):
    document: str
    doc_type: str
    style: str
    word_count: int
    tokens_used: int
    metadata: Dict[str, Any]

class StatusResponse(BaseModel):
    embeddings_ready: bool
    collection_name: str
    message: str

class SessionStats(BaseModel):
    interaction_count: int
    total_tokens_used: int
    max_interactions: int
    max_tokens: int
    embeddings_ready: bool

# ==================== DATABASE FUNCTIONS ====================

def init_database():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect("chat_logs.db", timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id TEXT PRIMARY KEY,
                username TEXT,
                chat_session_id TEXT,
                session_id TEXT,
                timestamp TEXT,
                user_input TEXT,
                bot_response TEXT,
                tokens_used INTEGER,
                is_flagged BOOLEAN,
                flag_reason TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database init error: {e}")

def log_interaction(username: str, chat_session_id: str, user_input: str, 
                   bot_response: str, tokens_used: int, is_flagged: bool = False, 
                   flag_reason: str = None):
    """Log chat interaction"""
    try:
        conn = sqlite3.connect("chat_logs.db", timeout=10)
        cursor = conn.cursor()
        
        log_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO chat_logs 
            (id, username, chat_session_id, session_id, timestamp, user_input, 
             bot_response, tokens_used, is_flagged, flag_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (log_id, username, chat_session_id, str(uuid.uuid4()), 
              timestamp, user_input, bot_response, tokens_used, is_flagged, flag_reason))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

def get_user_chat_sessions(username: str):
    """Get all chat sessions for a user"""
    try:
        conn = sqlite3.connect("chat_logs.db", timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                chat_session_id,
                MIN(timestamp) as first_msg_time,
                COUNT(*) as message_count,
                user_input
            FROM chat_logs
            WHERE username = ? AND chat_session_id IS NOT NULL
            GROUP BY chat_session_id
            ORDER BY first_msg_time DESC
        """, (username,))
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            chat_id, timestamp, count, first_input = row
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%b %d, %Y %I:%M %p")
            except:
                date_str = timestamp
            
            preview = (first_input[:40] + "...") if first_input and len(first_input) > 40 else (first_input or "Empty chat")
            
            sessions.append({
                "id": chat_id,
                "date": date_str,
                "count": count,
                "preview": preview
            })
        
        return sessions
    except Exception as e:
        logger.error(f"Error loading chat sessions: {e}")
        return []

def get_chat_messages(chat_session_id: str):
    """Get all messages from a specific chat session"""
    try:
        conn = sqlite3.connect("chat_logs.db", timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_input, bot_response, tokens_used, is_flagged, flag_reason
            FROM chat_logs
            WHERE chat_session_id = ?
            ORDER BY timestamp ASC
        """, (chat_session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            user_input, bot_response, tokens, flagged, flag_reason = row
            
            messages.append({
                "role": "user",
                "content": user_input
            })
            messages.append({
                "role": "assistant",
                "content": bot_response,
                "tokens_used": tokens,
                "is_flagged": bool(flagged),
                "flag_reason": flag_reason
            })
        
        return messages
    except Exception as e:
        logger.error(f"Error loading chat messages: {e}")
        return []

def delete_chat_session_db(chat_session_id: str, username: str):
    """Delete a chat session"""
    try:
        conn = sqlite3.connect("chat_logs.db", timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM chat_logs 
            WHERE chat_session_id = ? AND username = ?
        """, (chat_session_id, username))
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        return deleted > 0
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        return False

# ==================== AUTO-EMBEDDING FUNCTIONS ====================

def check_collection_exists():
    """Check if embeddings collection exists"""
    try:
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not qdrant_url or not qdrant_api_key:
            return False
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if PERSISTENT_COLLECTION_NAME in collection_names:
            collection_info = client.get_collection(PERSISTENT_COLLECTION_NAME)
            return collection_info.points_count > 0
        
        return False
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        return False

def get_documents_from_folder():
    """Get all documents from static folder"""
    if not DOCUMENTS_FOLDER.exists():
        return []
    
    document_files = []
    for ext in ALLOWED_EXTENSIONS:
        pattern = str(DOCUMENTS_FOLDER / f"*.{ext}")
        files = glob.glob(pattern)
        document_files.extend(files)
    
    return document_files

def auto_embed_documents():
    """Auto-embed documents on startup"""
    try:
        if check_collection_exists():
            return True, "Embeddings already exist"
        
        document_files = get_documents_from_folder()
        if not document_files:
            return False, "No documents found"
        
        embeddings_mgr = EmbeddingsManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            qdrant_url=os.getenv('QDRANT_URL'),
            collection_name=PERSISTENT_COLLECTION_NAME,
            chunk_size=1200,
            chunk_overlap=300
        )
        
        processed = 0
        for doc_path in document_files:
            try:
                embeddings_mgr.create_embeddings(doc_path)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {e}")
        
        if processed > 0:
            return True, f"Auto-embedded {processed} documents"
        return False, "Failed to embed documents"
    except Exception as e:
        logger.error(f"Auto-embedding error: {e}")
        return False, str(e)

# ==================== DOCUMENT GENERATION ====================

def generate_pdf_document(content: str, doc_type: str, metadata: dict) -> BytesIO:
    """Generate PDF document"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#1e3a8a',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        fontName='Times-Roman',
        leading=14
    )
    
    story = []
    story.append(Paragraph(doc_type.upper(), title_style))
    story.append(Spacer(1, 0.2*inch))
    
    metadata_text = f"<b>Generated:</b> {metadata.get('generated_at', 'N/A')}<br/>"
    metadata_text += f"<b>Jurisdiction:</b> {metadata.get('jurisdiction', 'General')}"
    story.append(Paragraph(metadata_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
    
    footer_text = "<i>Generated by Cognitbotz AI Legal Platform</i>"
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(footer_text, ParagraphStyle(
        'Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER
    )))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_docx_document(content: str, doc_type: str, metadata: dict) -> BytesIO:
    """Generate DOCX document"""
    if not DOCX_AVAILABLE:
        return None
    
    doc = DocxDocument()
    
    doc.core_properties.title = doc_type
    doc.core_properties.author = "Cognitbotz AI Legal Platform"
    
    title = doc.add_heading(doc_type.upper(), 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    meta_para = doc.add_paragraph()
    meta_para.add_run('Generated: ').bold = True
    meta_para.add_run(metadata.get('generated_at', 'N/A'))
    
    doc.add_paragraph()
    doc.add_paragraph('_' * 80)
    doc.add_paragraph()
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line:
            para = doc.add_paragraph(line)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer_run = footer.add_run('Generated by Cognitbotz AI Legal Platform')
    footer_run.italic = True
    footer_run.font.size = Pt(9)
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# ==================== JWT & AUTH ====================

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(username: str = Depends(verify_token)):
    """Get current authenticated user"""
    if username not in USERS_DB:
        raise HTTPException(status_code=401, detail="User not found")
    return username

def get_or_create_session(username: str):
    """Get or create session for user"""
    if username not in active_sessions:
        active_sessions[username] = {
            "session_id": str(uuid.uuid4()),
            "chat_session_id": str(uuid.uuid4()),
            "chatbot_manager": None,
            "drafting_manager": None,
            "interaction_count": 0,
            "total_tokens_used": 0,
            "created_at": datetime.now()
        }
    return active_sessions[username]

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Cognitbotz AI Legal Platform API...")
    
    # Initialize database
    init_database()
    
    # Auto-embed documents
    logger.info("Checking for embeddings...")
    success, message = auto_embed_documents()
    if success:
        logger.info(f"✅ {message}")
    else:
        logger.warning(f"⚠️ {message}")

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cognitbotz AI Legal Platform API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """User login"""
    username = request.username
    password = request.password
    
    if username not in USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_data = USERS_DB[username]
    if password != user_data["password"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    access_token = create_access_token(data={"sub": username})
    
    # Initialize session
    get_or_create_session(username)
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user={
            "username": username,
            "full_name": user_data["full_name"],
            "role": user_data["role"]
        }
    )

@app.get("/api/status", response_model=StatusResponse)
async def get_status(username: str = Depends(get_current_user)):
    """Get embeddings status"""
    embeddings_ready = check_collection_exists()
    
    return StatusResponse(
        embeddings_ready=embeddings_ready,
        collection_name=PERSISTENT_COLLECTION_NAME,
        message="Embeddings ready" if embeddings_ready else "Embeddings not ready"
    )

@app.get("/api/session/stats", response_model=SessionStats)
async def get_session_stats(username: str = Depends(get_current_user)):
    """Get session statistics"""
    session = get_or_create_session(username)
    embeddings_ready = check_collection_exists()
    
    return SessionStats(
        interaction_count=session["interaction_count"],
        total_tokens_used=session["total_tokens_used"],
        max_interactions=MAX_INTERACTIONS_PER_SESSION,
        max_tokens=MAX_TOKENS_PER_SESSION,
        embeddings_ready=embeddings_ready
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, username: str = Depends(get_current_user)):
    """Send chat message and get response"""
    session = get_or_create_session(username)
    
    # Check limits
    if session["interaction_count"] >= MAX_INTERACTIONS_PER_SESSION:
        raise HTTPException(status_code=429, detail="Interaction limit reached")
    
    if session["total_tokens_used"] >= MAX_TOKENS_PER_SESSION:
        raise HTTPException(status_code=429, detail="Token limit reached")
    
    # Use provided chat_session_id or current one
    chat_session_id = request.chat_session_id or session["chat_session_id"]
    
    try:
        # Initialize chatbot if needed
        if not session["chatbot_manager"]:
            session["chatbot_manager"] = ChatbotManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                llm_temperature=0.3,
                max_tokens=3000,
                qdrant_url=os.getenv('QDRANT_URL'),
                collection_name=PERSISTENT_COLLECTION_NAME,
                retrieval_k=5,
                score_threshold=0.5,
                use_custom_llm=False,
                custom_llm_url=None,
                custom_llm_api_key=None,
                custom_llm_model_name=None
            )
        
        # Determine RAG usage and layman mode
        use_rag = True
        layman_mode = request.layman_mode or False
        
        if request.chatbot_mode == "General Chat":
            use_rag = False
        elif request.chatbot_mode == "Hybrid (Smart)":
            use_rag = check_collection_exists()
        elif request.chatbot_mode == "Layman Explanation":
            use_rag = False
            layman_mode = True
        
        # Get response
        response = session["chatbot_manager"].get_response(
            request.message,
            enable_content_filter=True,
            enable_pii_detection=True,
            use_rag=use_rag,
            layman_mode=layman_mode
        )
        
        tokens_used = response.get('tokens_used', 0)
        session["interaction_count"] += 1
        session["total_tokens_used"] += tokens_used
        
        # Log interaction
        log_interaction(
            username=username,
            chat_session_id=chat_session_id,
            user_input=request.message,
            bot_response=response.get('answer', ''),
            tokens_used=tokens_used,
            is_flagged=response.get('is_flagged', False),
            flag_reason=response.get('flag_reason')
        )
        
        return ChatResponse(
            answer=response.get('answer', 'No response'),
            sources=response.get('sources', []),
            tokens_used=tokens_used,
            is_flagged=response.get('is_flagged', False),
            flag_reason=response.get('flag_reason'),
            response_type=response.get('response_type', 'rag'),
            chat_session_id=chat_session_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def get_chat_history(username: str = Depends(get_current_user)):
    """Get user's chat history"""
    sessions = get_user_chat_sessions(username)
    return {"sessions": sessions}

@app.get("/api/chat/{chat_session_id}")
async def get_chat(chat_session_id: str, username: str = Depends(get_current_user)):
    """Get specific chat session messages"""
    messages = get_chat_messages(chat_session_id)
    return {"messages": messages, "chat_session_id": chat_session_id}

@app.delete("/api/chat/{chat_session_id}")
async def delete_chat(chat_session_id: str, username: str = Depends(get_current_user)):
    """Delete a chat session"""
    success = delete_chat_session_db(chat_session_id, username)
    if success:
        return {"message": "Chat deleted successfully"}
    raise HTTPException(status_code=404, detail="Chat not found")

@app.post("/api/chat/new")
async def new_chat(username: str = Depends(get_current_user)):
    """Start a new chat session"""
    session = get_or_create_session(username)
    new_chat_id = str(uuid.uuid4())
    session["chat_session_id"] = new_chat_id
    session["interaction_count"] = 0
    
    return {"chat_session_id": new_chat_id, "message": "New chat started"}

@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    username: str = Depends(get_current_user)
):
    """Upload and process documents
    
    Note: Documents in the /documents folder are auto-embedded on startup.
    This endpoint is for adding additional documents dynamically.
    """
    processed = []
    failed = []
    
    try:
        embeddings_mgr = EmbeddingsManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            qdrant_url=os.getenv('QDRANT_URL'),
            collection_name=PERSISTENT_COLLECTION_NAME,
            chunk_size=1200,
            chunk_overlap=300
        )
        
        for file in files:
            # Validate file
            if file.size > MAX_FILE_SIZE:
                failed.append({"file": file.filename, "reason": "File too large"})
                continue
            
            ext = file.filename.split('.')[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                failed.append({"file": file.filename, "reason": "Invalid file type"})
                continue
            
            # Save temporarily and process
            try:
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, file.filename)
                
                with open(temp_path, 'wb') as f:
                    content = await file.read()
                    f.write(content)
                
                embeddings_mgr.create_embeddings(temp_path)
                processed.append(file.filename)
                
                # Cleanup
                os.remove(temp_path)
                os.rmdir(temp_dir)
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {e}")
                failed.append({"file": file.filename, "reason": str(e)})
        
        return {
            "processed": processed,
            "failed": failed,
            "message": f"Processed {len(processed)} documents"
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drafting/generate", response_model=DraftingResponse)
async def generate_document(
    request: DraftingRequest,
    username: str = Depends(get_current_user)
):
    """Generate legal document"""
    session = get_or_create_session(username)
    
    try:
        # Initialize drafting manager if needed
        if not session["drafting_manager"]:
            session["drafting_manager"] = LegalDraftingManager(
                llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.3,
                max_tokens=12000,
                user_preferences={}
            )
        
        # Generate document
        result = session["drafting_manager"].generate_document(
            doc_type=request.doc_type,
            prompt=request.requirements,
            style=request.style,
            length=request.length,
            clauses=request.clauses,
            special_provisions=request.special_provisions or ""
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        tokens_used = result.get('tokens_used', 0)
        session["total_tokens_used"] += tokens_used
        
        # Log interaction
        log_interaction(
            username=username,
            chat_session_id=session["chat_session_id"],
            user_input=request.requirements,
            bot_response=result.get('document', '')[:1000],
            tokens_used=tokens_used
        )
        
        return DraftingResponse(
            document=result.get('document', ''),
            doc_type=result.get('doc_type', ''),
            style=result.get('style', ''),
            word_count=result.get('word_count', 0),
            tokens_used=tokens_used,
            metadata=result.get('metadata', {})
        )
        
    except Exception as e:
        logger.error(f"Drafting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drafting/download/{format}")
async def download_document(
    format: str,
    content: str = Form(...),
    doc_type: str = Form(...),
    username: str = Depends(get_current_user)
):
    """Download generated document in specified format"""
    
    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "jurisdiction": "General"
    }
    
    if format == "txt":
        return StreamingResponse(
            iter([content]),
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={doc_type.replace(' ', '_')}.txt"
            }
        )
    
    elif format == "pdf":
        if not REPORTLAB_AVAILABLE:
            raise HTTPException(status_code=400, detail="PDF generation not available")
        
        pdf_buffer = generate_pdf_document(content, doc_type, metadata)
        if not pdf_buffer:
            raise HTTPException(status_code=500, detail="PDF generation failed")
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={doc_type.replace(' ', '_')}.pdf"
            }
        )
    
    elif format == "docx":
        if not DOCX_AVAILABLE:
            raise HTTPException(status_code=400, detail="DOCX generation not available")
        
        docx_buffer = generate_docx_document(content, doc_type, metadata)
        if not docx_buffer:
            raise HTTPException(status_code=500, detail="DOCX generation failed")
        
        return StreamingResponse(
            docx_buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename={doc_type.replace(' ', '_')}.docx"
            }
        )
    
    else:
        raise HTTPException(status_code=400, detail="Invalid format")

@app.post("/api/session/reset")
async def reset_session(username: str = Depends(get_current_user)):
    """Reset user session"""
    if username in active_sessions:
        del active_sessions[username]
    
    get_or_create_session(username)
    return {"message": "Session reset successfully"}

@app.post("/api/auth/logout")
async def logout(username: str = Depends(get_current_user)):
    """Logout user"""
    if username in active_sessions:
        del active_sessions[username]
    
    return {"message": "Logged out successfully"}

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    embeddings_ready = check_collection_exists()
    
    return {
        "status": "healthy",
        "embeddings_ready": embeddings_ready,
        "collection": PERSISTENT_COLLECTION_NAME,
        "timestamp": datetime.now().isoformat()
    }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")