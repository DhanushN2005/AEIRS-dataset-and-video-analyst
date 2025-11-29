
from typing import Optional, Dict, List
import os

class LLMExplainer:
    def __init__(self, provider: str = "openai", model: str = "gpt-4-turbo-preview"):
        self.provider = provider
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)
            except:
                return None
        elif self.provider == "anthropic" and self.api_key:
            try:
                import anthropic # type: ignore
                return anthropic.Anthropic(api_key=self.api_key)
            except:
                return None
        return None
    
    def explain_anomaly(self, anomaly_data: Dict) -> str:
        """Generate natural language explanation for anomaly"""
        prompt = f"""
        Analyze this anomaly and provide a clear, concise explanation:
        
        Feature: {anomaly_data.get('feature')}
        Value: {anomaly_data.get('value')}
        Expected: {anomaly_data.get('expected_value')}
        Deviation: {anomaly_data.get('deviation')}
        Detection Method: {anomaly_data.get('method')}
        
        Provide:
        1. What happened (1-2 sentences)
        2. Possible causes (2-3 bullet points)
        3. Recommended actions (2-3 bullet points)
        
        Keep it business-friendly and actionable.
        """
        
        return self._generate_response(prompt)
    
    def explain_correlation(self, correlation_data: Dict) -> str:
        """Explain correlation relationship"""
        prompt = f"""
        Explain this correlation finding:
        
        Feature 1: {correlation_data.get('feature1')}
        Feature 2: {correlation_data.get('feature2')}
        Correlation: {correlation_data.get('pearson_correlation')}
        Strength: {correlation_data.get('strength')}
        
        Provide a business interpretation and what this means for decision-making.
        """
        
        return self._generate_response(prompt)
    
    def generate_executive_summary(self, analysis_results: Dict) -> str:
        """Generate executive summary of entire analysis"""
        prompt = f"""
        Create an executive summary for this data analysis:
        
        Datasets Analyzed: {analysis_results.get('datasets_count')}
        Anomalies Detected: {analysis_results.get('anomalies_count')}
        Key Correlations: {analysis_results.get('correlations_count')}
        SDGs Identified: {analysis_results.get('sdg_count')}
        
        Key Findings:
        {self._format_key_findings(analysis_results.get('key_findings', []))}
        
        Provide:
        1. Executive Overview (2-3 sentences)
        2. Top 3 Insights
        3. Top 3 Recommended Actions
        4. Risk Assessment
        
        Format for C-level audience.
        """
        
        return self._generate_response(prompt)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using configured LLM"""
        if not self.client:
            return self._fallback_explanation(prompt)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert providing clear, actionable business insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        
        except Exception as e:
            return self._fallback_explanation(prompt)
    
    def _fallback_explanation(self, prompt: str) -> str:
        """Provide rule-based explanation when LLM unavailable"""
        return "LLM service unavailable. Analysis completed using statistical methods. Review visualizations and metrics for detailed insights."
    
    def _format_key_findings(self, findings: List[Dict]) -> str:
        """Format findings for prompt"""
        return "\n".join([f"- {f.get('title')}: {f.get('description')}" for f in findings[:5]])

