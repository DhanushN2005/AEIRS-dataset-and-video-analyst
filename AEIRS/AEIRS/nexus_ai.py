import os
import warnings
import base64
import json
from typing import Dict, Optional, List

# LangChain & AI Libraries
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Ensure this API key is correct
os.environ["GROQ_API_KEY"] = "..."

class NexusBot:
    def __init__(self):
        self.index_path = "nexus_memory_index"
        print("🔵 NEXUS: Initializing Neural Core...")
        
        # 1. Embeddings (Local - Fast & Free)
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        
        # 2. Vector Memory (RAG)
        self._load_memory()
            
        # 3. LLM Setup (Text)
        # Using the intelligent 70b model for text
        self.llm_text = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        
        # 4. Vision Model (Safeguarded)
        # Using 11b as it is currently the stable vision model
        self.llm_vision = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)


    def _load_memory(self):
        """Load or create the vector database"""
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embedding_function, 
                    allow_dangerous_deserialization=True
                )
                print("✅ NEXUS: Cognitive Memory Loaded.")
            except:
                self.vector_store = self._create_empty_memory()
        else:
            self.vector_store = self._create_empty_memory()
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

    def _create_empty_memory(self):
        """Create initial empty memory to prevent crash"""
        return FAISS.from_documents(
            [Document(page_content="NEXUS System Online. Awaiting data inputs.", metadata={"source": "system"})], 
            self.embedding_function
        )

    # =========================================================================
    # 🧠 CORE: INTELLIGENT INGESTION
    # =========================================================================
    def store_analysis(self, filename: str, analysis: Dict):
        """
        Ingests detailed project analysis into memory.
        """
        docs = []
        
        # Comprehensive Master Report
        report_content = [f"📄 **ANALYSIS REPORT: {filename}**"]
        
        # Overview
        report_content.append(f"**Overview:** {analysis['summary']['rows']} rows, {analysis['summary']['columns']} columns.")
        report_content.append(f"**Health:** {analysis.get('anomaly_count', 0)} Anomalies | {analysis.get('correlation_count', 0)} Correlations.")
        
        # Key Statistics
        if analysis.get('statistics'):
            report_content.append("\n📊 **Key Statistics:**")
            for col, stats in list(analysis['statistics'].items())[:5]: 
                report_content.append(f"- {col}: Mean {stats.get('mean', 0):.2f}, Max {stats.get('max', 0):.2f}")

        # Critical Anomalies
        if analysis.get('anomaly_details'):
            report_content.append("\n⚠️ **Critical Anomalies Detected:**")
            for a in analysis['anomaly_details'][:10]: 
                report_content.append(f"- Column '{a['column']}' at Row {a['index']}: Value {a['value']} (Deviation: {a['deviation']:.1f}σ)")

        # Strategic Correlations
        if analysis.get('correlations'):
            report_content.append("\n🔗 **Key Drivers (Correlations):**")
            for c in analysis['correlations'][:10]:
                report_content.append(f"- Strong link between '{c['feature1']}' and '{c['feature2']}' ({c['strength'].upper()}: {c['correlation']:.2f})")

        full_text = "\n".join(report_content)
        docs.append(Document(page_content=full_text, metadata={"source": filename, "type": "full_report"}))

        if docs:
            self.vector_store.add_documents(docs)
            self.vector_store.save_local(self.index_path)
            print(f"💾 NEXUS: Deep analysis for {filename} memorized.")

    # =========================================================================
    # 🗣️ FEATURE 1: PRO-LEVEL CHAT
    # =========================================================================
    def chat(self, user_query: str, target_language: str = "English") -> str:
        # 1. Retrieve Rich Context
        docs = self.retriever.invoke(user_query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 2. Advanced Prompt Engineering
        template = """
        You are NEXUS, an elite AI Data Consultant for AEIRS (Autonomous Enterprise Intelligence & Reporting System) Enterprise.
        Your goal is to provide insightful, structured, and professional answers.

        CONTEXT FROM DATABASE:
        {context}

        USER QUERY: {question}

        GUIDELINES:
        1. **Structure:** Use Bold Headers, Bullet points, and clear sections.
        2. **Insight:** Don't just list numbers. Explain *what they mean* for the business.
        3. **Missing Data:** If context is empty, use general knowledge politely.
        4. **Translation:** If the user asks to "Translate to [Language]", output ONLY the translated response. 
           Otherwise, respond in {language}.

        NEXUS RESPONSE:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm_text | StrOutputParser()
        
        try:
            return chain.invoke({
                "context": context_text,
                "question": user_query,
                "language": target_language
            })
        except Exception as e:
            return f"⚠️ NEXUS Brain Error: {str(e)}"

    # =========================================================================
    # 👁️ FEATURE 2: VISUAL CORTEX
    # =========================================================================
    def analyze_image(self, image_path: str, prompt: str = "Analyze this image.") -> str:
        """
        Robust Image Analysis with Error Handling
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                    },
                ]
            )
            response = self.llm_vision.invoke([message])
            return response.content
            
        except Exception as e:
            # ERROR HANDLER: If vision model is dead, return a polite message
            error_msg = str(e)
            if "model_decommissioned" in error_msg:
                return "⚠️ **Vision System Update:** The current vision model is currently offline. Please try text-based analysis."
            return f"❌ NEXUS Vision Error: {error_msg}"

# Initialize System for import
nexus_core = NexusBot()