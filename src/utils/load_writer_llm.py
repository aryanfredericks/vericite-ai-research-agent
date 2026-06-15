from langchain_groq import ChatGroq
import os

class LoadWriterLLM:
    def __init__(self, model_name : str):
        self.model_name = model_name
        self.llm = self._build_model()
        
    def _build_model(self):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        return ChatGroq(model=self.model_name, api_key=api_key)
    
    def get_model(self):
        return self.llm