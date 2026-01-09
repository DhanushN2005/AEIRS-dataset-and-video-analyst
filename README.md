# AEIRS – Autonomous Enterprise Intelligence & Reporting System

AEIRS is a unified AI-powered analysis platform designed to extract insights from videos, audio, images, and tabular datasets.

It combines LLM reasoning, automated analytics, and interactive visualization into a single modular system.

## 🚀 Core Modules
## 🎥 SnapSum — Video & Audio Intelligence

Analyze video/audio content with AI-driven understanding.

### Capabilities

- Automatic transcription
- Contextual summarization
- Sentiment analysis with visual indicators
- Key-point extraction
- Conversational Q&A with the media content

### Supported Formats

- MP4, MOV
- MP3, WAV

## 📊 AEIDS — Automated Enterprise Intelligence for Datasets

AI-powered exploratory data analysis for structured data.

### Features

- CSV / Excel upload
- Automated EDA (statistics, correlations, anomalies)
- Interactive visualizations
- Insight extraction and export (JSON / HTML reports)
- Persistent analysis history

## 🤖 NEXUS AI — Intelligent Data Copilot

An embedded AI assistant that understands your analysis context.

### Capabilities

- Natural language Q&A over dataset insights
- RAG over stored analysis outputs
- Optional OCR & image understanding via vision models
- Context-aware reasoning inside the analysis workflow

## 🧠 System Architecture
Frontend (HTML/CSS/JS)

        |
        
        v
        
FastAPI Backend (Modular Routers)

        |
        
        ├── SnapSum Pipeline (Media Analysis)
        
        ├── AEIDS Pipeline (Dataset Analysis)
        
        ├── NEXUS AI (RAG + Reasoning)
        
        |
        
Vector DB (Pinecone / FAISS)

        |
        
SQLite (Metadata & History)

## 🛠 Tech Stack

### Backend

- FastAPI
- Uvicorn
- Groq LLMs (Text + Vision)
- Pinecone / FAISS (Vector Search)
- pandas, numpy
- SQLite

### Frontend

- HTML, CSS, JavaScript
- Font Awesome
- Responsive UI
- Dark / Light theme support

## 📂 Project Structure
AEIRS-DATA-and-VIDEO-REPORTER/

│

├── AEIRS/

│   ├── analyser.py

│   ├── detectors.py

│   ├── pipeline.py

│   ├── llm_explainer.py

│   ├── models.py

│   ├── auth.py

│   └── app.py

│

├── frontend/

│   ├── video_analysis.html

│   ├── new.html

│   └── login.html

│

├── uploads/

├── requirements.txt

└── README.md

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

cd AEIRS-DATA-and-VIDEO-REPORTER/AEIRS

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Configure Environment Variables

Create a .env file or export variables:

GROQ_API_KEY=your_key_here

PINECONE_API_KEY=your_key_here

### 4️⃣ Run Application
uvicorn app:app --reload


### Access:

http://localhost:8000 → Login

/video-analysis

/dataset-analysis

## 📌 Use Cases

- Enterprise video audits
- Interview & meeting analysis
- Automated business intelligence
- Dataset exploration for analysts
- AI-assisted reporting workflows

## 🔒 Security Notice

API keys must never be committed to source code or documentation.

Always use environment variables or secret managers.

## 📈 Future Enhancements

- Role-based authentication
- Cloud deployment (Docker + AWS/GCP)
- Multi-user project spaces
- Streaming video analysis
- Model switching support


