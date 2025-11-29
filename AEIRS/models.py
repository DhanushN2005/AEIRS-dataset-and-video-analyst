
# ============================================================================
# 3. models.py - Database Models
# ============================================================================
MODELS_PY = """
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default="user")  # user, admin, analyst
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    datasets = relationship("Dataset", back_populates="owner")
    analyses = relationship("Analysis", back_populates="user")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    file_type = Column(String)
    rows = Column(Integer)
    columns = Column(Integer)
    metadata = Column(JSON)
    owner_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="datasets")
    analyses = relationship("Analysis", back_populates="dataset")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("datasets.id"))
    user_id = Column(String, ForeignKey("users.id"))
    status = Column(String, default="pending")  # pending, running, completed, failed
    config = Column(JSON)
    results = Column(JSON)
    anomalies_count = Column(Integer, default=0)
    insights_count = Column(Integer, default=0)
    sdg_count = Column(Integer, default=0)
    report_path = Column(String)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    dataset = relationship("Dataset", back_populates="analyses")
    user = relationship("User", back_populates="analyses")
    anomalies = relationship("Anomaly", back_populates="analysis")
    insights = relationship("Insight", back_populates="analysis")

class Anomaly(Base):
    __tablename__ = "anomalies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"))
    timestamp = Column(DateTime)
    feature = Column(String)
    value = Column(Float)
    expected_value = Column(Float)
    deviation = Column(Float)
    severity = Column(String)  # low, medium, high, critical
    method = Column(String)
    explanation = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("Analysis", back_populates="anomalies")

class Insight(Base):
    __tablename__ = "insights"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"))
    type = Column(String)  # anomaly, correlation, causal, sdg, geo
    severity = Column(String)
    title = Column(String)
    description = Column(Text)
    confidence = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("Analysis", back_populates="insights")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("analyses.id"))
    type = Column(String)  # email, slack, webhook
    status = Column(String)  # sent, failed, pending
    message = Column(Text)
    sent_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
"""
