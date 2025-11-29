from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
from datetime import datetime
import uvicorn

app = FastAPI(title="AEIDS Enterprise", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("frontend", exist_ok=True)

# In-memory storage
uploaded_files = []
analysis_results = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend"""
    try:
        with open("frontend/new.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AEIDS Enterprise</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .card {
                    background: white;
                    color: #333;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                }
                h1 { color: #667eea; }
                a {
                    color: #667eea;
                    text-decoration: none;
                    font-weight: bold;
                }
                .btn {
                    display: inline-block;
                    padding: 10px 20px;
                    background: #667eea;
                    color: white;
                    border-radius: 5px;
                    margin: 10px 5px;
                }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🧠 AEIDS Server Running ✅</h1>
                <p>The backend is running successfully!</p>
                <h3>⚠️ Frontend Not Found</h3>
                <p>Create <code>frontend/new.html</code> to see the full interface.</p>
                <h3>📚 Available Resources:</h3>
                <a href="/docs" class="btn">📖 API Documentation</a>
                <a href="/health" class="btn">💚 Health Check</a>
                <h3>🔌 API Endpoints:</h3>
                <ul>
                    <li><code>POST /api/upload</code> - Upload dataset</li>
                    <li><code>GET /api/files</code> - List uploaded files</li>
                    <li><code>POST /api/analyze/{id}</code> - Run analysis</li>
                    <li><code>GET /api/analysis/{id}</code> - Get results</li>
                    <li><code>GET /api/stats</code> - System statistics</li>
                </ul>
            </div>
        </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload dataset file"""
    try:
        # Save file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read file to get metadata
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise HTTPException(400, "Unsupported file format. Use CSV or Excel files.")
        except UnicodeDecodeError:
            # Try different encodings
            df = pd.read_csv(file_path, encoding='latin-1')
        
        file_info = {
            "id": len(uploaded_files) + 1,
            "filename": file.filename,
            "path": file_path,
            "size": len(content),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "uploaded_at": datetime.now().isoformat()
        }
        
        uploaded_files.append(file_info)
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "file": file_info
        }
    
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/api/files")
async def get_files():
    """Get list of uploaded files"""
    return {"files": uploaded_files, "count": len(uploaded_files)}

@app.post("/api/analyze/{file_id}")
async def analyze_file(file_id: int):
    """Run analysis pipeline on uploaded file"""
    try:
        # Find file
        file_info = next((f for f in uploaded_files if f["id"] == file_id), None)
        if not file_info:
            raise HTTPException(404, "File not found")
        
        # Load data with encoding handling
        try:
            if file_info["filename"].endswith('.csv'):
                df = pd.read_csv(file_info["path"], encoding='utf-8')
            else:
                df = pd.read_excel(file_info["path"])
        except UnicodeDecodeError:
            df = pd.read_csv(file_info["path"], encoding='latin-1')
        
        # Simple analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            return {
                "file_id": file_id,
                "status": "completed",
                "warning": "No numeric columns found for analysis",
                "summary": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": 0
                }
            }
        
        # Calculate basic stats
        stats = {}
        for col in numeric_cols[:10]:  # Limit to 10 columns
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }
        
        # Detect simple anomalies (values > 3 std devs)
        anomalies = {}
        anomaly_details = []
        
        for col in numeric_cols[:10]:
            mean = df[col].mean()
            std = df[col].std()
            threshold = 3
            
            if std > 0:  # Avoid division by zero
                anomaly_mask = (df[col] - mean).abs() > (threshold * std)
                anomaly_count = int(anomaly_mask.sum())
                anomaly_indices = anomaly_mask[anomaly_mask].index.tolist()[:10]
                
                anomalies[col] = {
                    "count": anomaly_count,
                    "indices": anomaly_indices
                }
                
                # Add detailed anomaly info
                for idx in anomaly_indices[:5]:  # Top 5 per column
                    anomaly_details.append({
                        "column": col,
                        "index": int(idx),
                        "value": float(df[col].iloc[idx]),
                        "expected": float(mean),
                        "deviation": float((df[col].iloc[idx] - mean) / std)
                    })
        
        # Calculate correlations
        correlations = []
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5 and not pd.isna(corr_val):
                        correlations.append({
                            "feature1": numeric_cols[i],
                            "feature2": numeric_cols[j],
                            "correlation": float(corr_val),
                            "strength": "strong" if abs(corr_val) > 0.7 else "moderate"
                        })
        
        # Sort correlations by strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        result = {
            "file_id": file_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "column_names": df.columns.tolist()
            },
            "statistics": stats,
            "anomalies": anomalies,
            "anomaly_details": anomaly_details,
            "correlations": correlations[:20],  # Top 20 correlations
            "anomaly_count": sum(v["count"] for v in anomalies.values()),
            "correlation_count": len(correlations)
        }
        
        analysis_results[file_id] = result
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.get("/api/analysis/{file_id}")
async def get_analysis(file_id: int):
    """Get analysis results"""
    if file_id not in analysis_results:
        raise HTTPException(404, "Analysis not found. Run analysis first.")
    return analysis_results[file_id]

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
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
        "version": "1.0.0"
    }

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: int):
    """Delete uploaded file"""
    global uploaded_files
    file_info = next((f for f in uploaded_files if f["id"] == file_id), None)
    
    if not file_info:
        raise HTTPException(404, "File not found")
    
    # Delete physical file
    try:
        if os.path.exists(file_info["path"]):
            os.remove(file_info["path"])
    except Exception as e:
        pass
    
    # Remove from list
    uploaded_files = [f for f in uploaded_files if f["id"] != file_id]
    
    # Remove analysis results
    if file_id in analysis_results:
        del analysis_results[file_id]
    
    return {"success": True, "message": "File deleted"}

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("🚀 AEIDS Enterprise Server Starting...")
    print("=" * 60)
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("💡 Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)