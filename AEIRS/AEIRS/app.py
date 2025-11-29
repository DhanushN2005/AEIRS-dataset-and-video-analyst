import os
import io 
import json
import pandas as pd
import uuid
import sqlite3
import tempfile
from groq import Groq
from datetime import datetime
from typing import List, Dict, Any
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, APIRouter, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from nexus_ai import nexus_core
import math

# ============ CONFIG ============

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "...")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "...")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # e.g. 'us-east-1-aws'
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index")

DB_PATH = os.getenv("INTERVIEW_BOT_DB", "interview_bot.db")

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# ============ PINECONE INIT ============

def init_pinecone():
    if not PINECONE_API_KEY:
        print("WARNING: PINECONE_API_KEY not set, vector features will not work.")
        return None

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print("ERROR: Could not initialize Pinecone client:", e)
        return None

    try:
        listed = pc.list_indexes()
        if hasattr(listed, "names"):
            existing_index_names = listed.names()
        else:
            existing_index_names = [idx["name"] for idx in listed]

        if PINECONE_INDEX_NAME not in existing_index_names:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,        # our simple fake embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )

        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Pinecone: connected to index '{PINECONE_INDEX_NAME}'")
        return index
    except Exception as e:
        print(f"WARNING: Could not connect/create Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        return None

pinecone_index = init_pinecone()

# ============ DB HELPERS ============

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    cur = conn.cursor()

    # Existing table for interviews
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interviews (
        id TEXT PRIMARY KEY,
        participant TEXT,
        location TEXT,
        date TEXT,
        created_at TEXT,
        summary TEXT,
        sentiment TEXT,
        key_points TEXT
    );
    """)

    # NEW: table for uploaded datasets (AEIDS)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        path TEXT,
        size INTEGER,
        rows INTEGER,
        columns INTEGER,
        column_names TEXT,
        uploaded_at TEXT
    );
    """)

    # NEW: table for dataset analysis results (stored as JSON)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dataset_analyses (
        file_id INTEGER PRIMARY KEY,
        analysis_json TEXT,
        FOREIGN KEY(file_id) REFERENCES datasets(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
# ============ FASTAPI SETUP ============

app = FastAPI(title="SnapSum (Groq + Pinecone)")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(__file__)

# app.py is in AEIDS/AEIDS/app.py; project root is one level up
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

AEIDS_FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
AEIDS_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
AEIDS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(AEIDS_UPLOAD_DIR, exist_ok=True)
os.makedirs(AEIDS_OUTPUT_DIR, exist_ok=True)

# In-memory storage for dataset analysis
uploaded_files: list[dict] = []
analysis_results: dict[int, dict] = {}

# ---------- DATASET PERSISTENCE HELPERS ----------

def load_dataset_state_from_db():
    """
    Load persisted AEIDS datasets + analyses into the in-memory
    uploaded_files and analysis_results structures.
    """
    global uploaded_files, analysis_results

    conn = get_db_conn()
    cur = conn.cursor()

    # Load dataset metadata
    cur.execute("""
        SELECT id, filename, path, size, rows, columns, column_names, uploaded_at
        FROM datasets
        ORDER BY id;
    """)
    uploaded_files = []
    for row in cur.fetchall():
        try:
            col_names = json.loads(row["column_names"]) if row["column_names"] else []
        except json.JSONDecodeError:
            col_names = []
        uploaded_files.append({
            "id": row["id"],
            "filename": row["filename"],
            "path": row["path"],
            "size": row["size"],
            "rows": row["rows"],
            "columns": row["columns"],
            "column_names": col_names,
            "uploaded_at": row["uploaded_at"],
        })

    # Load analysis results
    cur.execute("SELECT file_id, analysis_json FROM dataset_analyses;")
    analysis_results = {}
    for row in cur.fetchall():
        try:
            analysis_results[row["file_id"]] = json.loads(row["analysis_json"])
        except json.JSONDecodeError:
            # Skip corrupted rows instead of crashing
            continue

    conn.close()


def persist_dataset_file(file_info: dict):
    """Insert / update dataset metadata in the DB."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO datasets
            (id, filename, path, size, rows, columns, column_names, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            file_info["id"],
            file_info["filename"],
            file_info["path"],
            file_info.get("size"),
            file_info.get("rows"),
            file_info.get("columns"),
            json.dumps(file_info.get("column_names", [])),
            file_info.get("uploaded_at"),
        ),
    )
    conn.commit()
    conn.close()


def persist_dataset_analysis(file_id: int, analysis: dict):
    """Insert / update analysis result JSON in the DB."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO dataset_analyses (file_id, analysis_json)
        VALUES (?, ?);
        """,
        (file_id, json.dumps(analysis)),
    )
    conn.commit()
    conn.close()


def delete_dataset_from_db(file_id: int):
    """Delete dataset + its analysis from DB."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM dataset_analyses WHERE file_id = ?;", (file_id,))
    cur.execute("DELETE FROM datasets          WHERE id      = ?;", (file_id,))
    conn.commit()
    conn.close()

# ============ API MODELS ============

class InterviewCreateResponse(BaseModel):
    interview_id: str
    participant: str
    location: str
    date: str
    summary: str
    sentiment: str
    key_points: List[str]

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
class OptimusChatRequest(BaseModel):
    message: str
    file_id: int | None = None  # optional link to an uploaded dataset


class OptimusChatResponse(BaseModel):
    reply: str


class OptimusImageReportResponse(BaseModel):
    report: str


# ============ LLM CLIENT (GROQ) ============

class LLMClient:
    """
    Real LLM via Groq (llama-3.1-8b-instant).
    """
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.1-8b-instant"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        content = self.chat(messages)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start:end+1])
            raise

llm_client = LLMClient(groq_client)

# ============ ASR (GROQ WHISPER) ============

class ASRClient:
    """
    Real ASR using Groq Whisper API (whisper-large-v3).
    No local ffmpeg, no local model.
    """
    def __init__(self, client: Groq):
        self.client = client

    def transcribe_bytes(self, file_bytes: bytes, filename: str) -> str:
        # Use in-memory bytes as file-like object
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = filename or "audio.mp4"

        result = self.client.audio.transcriptions.create(
            file=file_obj,
            model="whisper-large-v3"
        )

        text = getattr(result, "text", "") or ""
        if not text.strip():
            text = "Transcript could not be generated."

        return text

asr_client = ASRClient(groq_client)

# ============ VECTOR AGENT (FAKE EMBEDDINGS + PINECONE) ============

class VectorAgent:
    """
    Uses a simple deterministic embedding (hash-based).
    Not true semantic embeddings, but enough to demo RAG architecture
    without needing another embedding API.
    """
    def __init__(self, index, dim: int = 1024):
        self.index = index
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        b = text.encode("utf-8", errors="ignore")
        for i, bt in enumerate(b):
            vec[i % self.dim] += float(bt)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def chunk_text(self, text: str, max_chars: int = 1000) -> List[str]:
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
        return chunks or [text]

    def index_interview(self, interview_id: str, transcript: str, metadata: Dict[str, Any]):
        if self.index is None:
            return
        chunks = self.chunk_text(transcript)
        vectors = []
        for i, chunk in enumerate(chunks):
            emb = self._embed(chunk)
            vectors.append({
                "id": f"{interview_id}_chunk_{i}",
                "values": emb,
                "metadata": {
                    "interview_id": interview_id,
                    **metadata,
                    "chunk_index": i,
                    "text": chunk
                }
            })
        self.index.upsert(vectors)

    def search(self, interview_id: str, question: str, top_k: int = 5):
        if self.index is None:
            return []
        q_emb = self._embed(question)
        res = self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
            filter={"interview_id": {"$eq": interview_id}}
        )
        # Pinecone new client might return object with .matches
        if hasattr(res, "matches"):
            return res.matches
        return res["matches"]

vector_agent = VectorAgent(pinecone_index)

# ============ SCHEMA AGENT (SIMPLE) ============

class SchemaAgent:
    """
    Simple schema agent: just ensures summary/sentiment/key_points columns
    and writes values. (Currently unused because we write directly on insert.)
    """
    def upsert_interview_record(self, interview_id: str, summary_payload: Dict[str, Any]):
        summary = summary_payload.get("summary")
        sentiment = summary_payload.get("sentiment")
        key_points = summary_payload.get("key_points") or []
        if not isinstance(key_points, list):
            key_points = [str(key_points)]
        key_points_str = "; ".join(key_points)

        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE interviews
            SET summary = ?, sentiment = ?, key_points = ?
            WHERE id = ?;
        """, (summary, sentiment, key_points_str, interview_id))
        conn.commit()
        conn.close()

schema_agent = SchemaAgent()

# ============ INTERVIEW AGENT ============

class InterviewSummary(BaseModel):
    interview_id: str
    participant: str
    location: str
    date: str
    summary: str
    sentiment: str
    key_points: List[str]


class InterviewAgent:
    def __init__(self, asr: ASRClient, llm: LLMClient):
        self.asr = asr
        self.llm = llm

    def process_video(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        # 1) Transcribe
        transcript = self.asr.transcribe_bytes(file_bytes, filename)

        # 2) Ask Groq to extract summary, sentiment, key points AND metadata
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant analyzing an interview transcript.\n"
                    "You MUST respond ONLY with valid JSON. No explanations.\n\n"
                    "Return a JSON object with EXACTLY these keys:\n"
                    "- summary: string\n"
                    "- sentiment: one of ['positive','neutral','negative']\n"
                    "- key_points: list of short strings\n"
                    "- participant: string (name of the person being interviewed, or 'Unknown')\n"
                    "- location: string (place of interview, or 'Unknown')\n"
                    "- date: string (ISO format if present in transcript, else 'Unknown')\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Transcript:\n\n"
                    f"{transcript}\n\n"
                    "Now generate the JSON with summary, sentiment, key_points, participant, location, date."
                )
            }
        ]

        result = self.llm.chat_json(messages)

        # Safety defaults
        summary = result.get("summary", "Summary not available.")
        sentiment = result.get("sentiment", "neutral")
        key_points = result.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = [str(key_points)]

        participant = result.get("participant") or "Unknown"
        location = result.get("location") or "Unknown"
        date = result.get("date") or "Unknown"

        return {
            "transcript": transcript,
            "summary": summary,
            "sentiment": sentiment,
            "key_points": key_points,
            "participant": participant,
            "location": location,
            "date": date,
        }

interview_agent = InterviewAgent(asr_client, llm_client)

# ============ QUERY AGENT (REAL ANSWER WITH CONTEXT) ============

class QueryAgent:
    def __init__(self, llm: LLMClient, vector_agent: VectorAgent):
        self.llm = llm
        self.vector_agent = vector_agent

    def ask(self, interview_id: str, question: str) -> str:
        # 1) Load summary / sentiment / key_points from the DB
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT summary, sentiment, key_points FROM interviews WHERE id = ?;",
            (interview_id,)
        )
        row = cur.fetchone()
        conn.close()

        summary = ""
        sentiment = ""
        key_points_list: List[str] = []

        if row:
            row_dict = dict(row)
            summary = row_dict.get("summary") or ""
            sentiment = row_dict.get("sentiment") or ""
            key_points_str = row_dict.get("key_points") or ""
            key_points_list = [kp.strip() for kp in key_points_str.split(";") if kp.strip()]

        # 2) Retrieve transcript chunks from Pinecone (if any)
        matches = self.vector_agent.search(interview_id, question, top_k=5)
        context_chunks = []
        for m in matches:
            meta = m["metadata"]
            text = meta.get("text", "")
            if text:
                context_chunks.append(text)

        transcript_context = "\n\n".join(context_chunks) if context_chunks else "No transcript chunks were retrieved."

        # 3) Build a rich context for the LLM (no f-strings to avoid backslash issues)
        key_points_block = (
            "- " + "\n- ".join(key_points_list)
            if key_points_list else "No key points stored."
        )

        context_block = (
            "SUMMARY:\n" + (summary or "No summary available.") + "\n\n"
            "SENTIMENT:\n" + (sentiment or "Unknown") + "\n\n"
            "KEY POINTS:\n" + key_points_block + "\n\n"
            "TRANSCRIPT EXCERPTS:\n" + transcript_context + "\n"
        )

        # 4) Ask Groq to answer using this context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Interview Bot. You answer questions about an interview.\n"
                    "You are given:\n"
                    "- A summary of the interview\n"
                    "- An overall sentiment\n"
                    "- A list of key points (these are effectively the main keywords/topics)\n"
                    "- Optional transcript excerpts retrieved via vector search\n\n"
                    "RULES:\n"
                    "1. Always use the summary and key points as your primary source of truth.\n"
                    "2. Use transcript excerpts to refine or add detail, but do not contradict the summary.\n"
                    "3. If the user asks for 'key words', 'keywords', 'key points', 'topics', or similar, "
                    "respond by listing concise keywords/phrases derived from the summary and key_points list.\n"
                    "4. Only say you are not sure if there is truly no information in the context about the question.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question: " + question + "\n\n"
                    "Interview context:\n" + context_block
                ),
            },
        ]

        answer = self.llm.chat(messages)
        return answer


query_agent = QueryAgent(llm_client, vector_agent)

# ============ ROUTES ============

@app.on_event("startup")
def on_startup():
    init_db()
    load_dataset_state_from_db()


# 🔹 1. LOGIN AT ROOT "/"
@app.get("/", response_class=HTMLResponse)
async def serve_login():
    login_path = os.path.join(BASE_DIR, "login.html")
    with open(login_path, "r", encoding="utf-8") as f:
        return f.read()

# 🔹 2. WELCOME DASHBOARD
@app.get("/welcome", response_class=HTMLResponse)
async def serve_welcome():
    welcome_path = os.path.join(BASE_DIR, "welcome.html")
    with open(welcome_path, "r", encoding="utf-8") as f:
        return f.read()

# 🔹 3. VIDEO ANALYSIS (SnapSum UI)
@app.get("/video-analysis", response_class=HTMLResponse)
async def serve_video_analysis():
    va_path = os.path.join(BASE_DIR, "video_analysis.html")
    with open(va_path, "r", encoding="utf-8") as f:
        return f.read()
    
# 🔹 4. Dataset ANALYSIS 
@app.get("/dataset-analysis", response_class=HTMLResponse)
async def dataset_analysis_page():
    """Serve the AEIDS dataset analysis UI."""
    html_path = os.path.join(AEIDS_FRONTEND_DIR, "new.html")
    if not os.path.exists(html_path):
        return HTMLResponse(
            "<h1>AEIDS UI not found</h1><p>Expected frontend/new.html</p>",
            status_code=500,
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/interviews", response_model=InterviewCreateResponse)
async def create_interview(
    file: UploadFile = File(...),
):
    # Read file bytes
    file_bytes = await file.read()
    interview_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    # Run interview agent (Groq) → get transcript, summary, sentiment, key_points, and metadata
    result = interview_agent.process_video(file_bytes, file.filename)

    participant = result["participant"]
    location = result["location"]
    date = result["date"]

    # Store row in DB
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO interviews (id, participant, location, date, created_at, summary, sentiment, key_points)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            interview_id,
            participant,
            location,
            date,
            created_at,
            result["summary"],
            result["sentiment"],
            "; ".join(result["key_points"]),
        )
    )
    conn.commit()
    conn.close()

    # Index transcript in Pinecone
    if pinecone_index is not None:
        vector_agent.index_interview(
            interview_id,
            result["transcript"],
            {
                "participant": participant,
                "location": location,
                "date": date
            }
        )

    return InterviewCreateResponse(
        interview_id=interview_id,
        participant=participant,
        location=location,
        date=date,
        summary=result["summary"],
        sentiment=result["sentiment"],
        key_points=result["key_points"],
    )

@app.get("/interviews/{interview_id}", response_model=InterviewSummary)
async def get_interview(interview_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM interviews WHERE id = ?;", (interview_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Interview not found")

    row_dict = dict(row)
    key_points_str = row_dict.get("key_points") or ""
    key_points = [kp.strip() for kp in key_points_str.split(";") if kp.strip()]

    return InterviewSummary(
        interview_id=row_dict["id"],
        participant=row_dict["participant"],
        location=row_dict["location"],
        date=row_dict["date"],
        summary=row_dict.get("summary") or "Summary not available.",
        sentiment=row_dict.get("sentiment") or "neutral",
        key_points=key_points,
    )

@app.post("/interviews/{interview_id}/ask", response_model=AskResponse)
async def ask_interview(interview_id: str, req: AskRequest):
    answer = query_agent.ask(interview_id, req.question)
    return AskResponse(answer=answer)

# ---------- AEIDS Dataset Analysis API (mounted under /dataset-analysis/api) ----------

aeids_router = APIRouter(prefix="/dataset-analysis/api", tags=["dataset-analysis"])


@aeids_router.get("/health")
async def aeids_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@aeids_router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset file (CSV / Excel)."""
    try:
        # save file
        file_path = os.path.join(AEIDS_UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # read file to get metadata
        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path, encoding="utf-8")
            elif file.filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            else:
                raise HTTPException(
                    400, "Unsupported file format. Use CSV or Excel files."
                )
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1")

        file_info = {
            "id": (max((f["id"] for f in uploaded_files), default=0) + 1),
            "filename": file.filename,
            "path": file_path,
            "size": len(content),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "uploaded_at": datetime.now().isoformat(),
        }

        uploaded_files.append(file_info)
        persist_dataset_file(file_info)

        return {
            "success": True,
            "message": "File uploaded successfully",
            "file": file_info,
        }

    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@aeids_router.get("/files")
async def get_files():
    """List uploaded files."""
    return {"files": uploaded_files, "count": len(uploaded_files)}


@aeids_router.post("/analyze/{file_id}")
async def analyze_file(file_id: int):
    """Run analysis pipeline on uploaded file."""
    try:
        # find file
        file_info = next((f for f in uploaded_files if f["id"] == file_id), None)
        if not file_info:
            raise HTTPException(404, "File not found")

        # load data
        try:
            if file_info["filename"].endswith(".csv"):
                df = pd.read_csv(file_info["path"], encoding="utf-8")
            else:
                df = pd.read_excel(file_info["path"])
        except UnicodeDecodeError:
            df = pd.read_csv(file_info["path"], encoding="latin-1")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if len(numeric_cols) == 0:
            result = {
                "file_id": file_id,
                "status": "completed",
                "warning": "No numeric columns found for analysis",
                "summary": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": 0,
                },
            }
            analysis_results[file_id] = result
            persist_dataset_analysis(file_id, result)
            return result

        # basic stats
        stats = {}
        for col in numeric_cols[:10]:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }

        # simple anomalies > 3σ
        anomalies = {}
        anomaly_details = []
        for col in numeric_cols[:10]:
            mean = df[col].mean()
            std = df[col].std()
            threshold = 3

            if std > 0:
                mask = (df[col] - mean).abs() > (threshold * std)
                count = int(mask.sum())
                idxs = mask[mask].index.tolist()[:10]

                anomalies[col] = {"count": count, "indices": idxs}

                for idx in idxs[:5]:
                    anomaly_details.append(
                        {
                            "column": col,
                            "index": int(idx),
                            "value": float(df[col].iloc[idx]),
                            "expected": float(mean),
                            "deviation": float((df[col].iloc[idx] - mean) / std),
                        }
                    )

        # correlations
        correlations = []
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and abs(corr_val) > 0.5:
                        correlations.append(
                            {
                                "feature1": numeric_cols[i],
                                "feature2": numeric_cols[j],
                                "correlation": float(corr_val),
                                "strength": "strong"
                                if abs(corr_val) > 0.7
                                else "moderate",
                            }
                        )

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        result = {
            "file_id": file_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "column_names": df.columns.tolist(),
            },
            "statistics": stats,
            "anomalies": anomalies,
            "anomaly_details": anomaly_details,
            "correlations": correlations[:20],
            "anomaly_count": sum(v["count"] for v in anomalies.values()),
            "correlation_count": len(correlations),
        }

        analysis_results[file_id] = result
        persist_dataset_analysis(file_id, result)
        try:
            print(f"🔌 NEXUS: learning from {file_info['filename']}...")
            nexus_core.store_analysis(file_info["filename"], result)
        except Exception as e:
            # Don't break the main flow if NEXUS has an issue
            print(f"⚠️ NEXUS ingestion warning: {e}")
        return result

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

# ---------- NEXUS AI BOT (Dataset Analysis) ----------

@aeids_router.post("/nexus/chat")
async def nexus_chat(
    query: str = Form(...),
    language: str = Form("English"),
    image: UploadFile | None = File(None),
):
    """
    Unified endpoint for the NEXUS Bot.
    Handles text questions, RAG over stored analyses, and optional image input.
    """
    response_text = ""

    # 1) Optional vision branch
    if image is not None:
        try:
            tmp_name = f"temp_vision_{image.filename}"
            tmp_path = os.path.join(AEIDS_UPLOAD_DIR, tmp_name)
            with open(tmp_path, "wb") as f:
                f.write(await image.read())

            vision_res = nexus_core.analyze_image(tmp_path, query)
            response_text += f"👁️ **Visual Analysis:**\n{vision_res}\n\n"

            try:
                os.remove(tmp_path)
            except Exception:
                pass
        except Exception as e:
            response_text += f"⚠️ Vision Error: {e}\n\n"

    # 2) Text + RAG
    try:
        text_res = nexus_core.chat(query, language)
    except Exception as e:
        text_res = f"⚠️ NEXUS error: {e}"

    response_text += text_res

    return {
        "status": "success",
        "agent": "NEXUS-v1",
        "response": response_text,
        "timestamp": datetime.now().isoformat(),
    }

@aeids_router.get("/analysis/{file_id}")
async def get_analysis(file_id: int):
    if file_id not in analysis_results:
        raise HTTPException(404, "Analysis not found. Run analysis first.")
    return analysis_results[file_id]


@aeids_router.get("/stats")
async def get_stats():
    total_rows = sum(f.get("rows", 0) for f in uploaded_files)
    return {
        "datasets": len(uploaded_files),
        "analyses": len(analysis_results),
        "total_rows": total_rows,
        "total_anomalies": sum(
            r.get("anomaly_count", 0) for r in analysis_results.values()
        ),
        "total_correlations": sum(
            r.get("correlation_count", 0) for r in analysis_results.values()
        ),
        "uptime": "Running",
        "version": "1.0.0",
    }


@aeids_router.delete("/files/{file_id}")
async def delete_dataset_file(file_id: int):
    """Delete uploaded file & its analysis."""
    global uploaded_files
    file_info = next((f for f in uploaded_files if f["id"] == file_id), None)
    if not file_info:
        raise HTTPException(404, "File not found")

    try:
        if os.path.exists(file_info["path"]):
            os.remove(file_info["path"])
    except Exception:
        pass

    uploaded_files = [f for f in uploaded_files if f["id"] != file_id]
    if file_id in analysis_results:
        del analysis_results[file_id]
    delete_dataset_from_db(file_id)
    return {"success": True, "message": "File deleted"}


app.include_router(aeids_router)

# ============ RUN (local dev) ============

if __name__ == "__main__":
    import uvicorn
    import sys
    print("🚀 AEIRS Enterprise Server Starting...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("💡 Health Check: http://localhost:8000/health")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)