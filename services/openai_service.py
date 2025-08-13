import asyncio
import openai
from typing import List, Dict, Any, Optional
from openai import AsyncAzureOpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class OpenAIService:
    def __init__(self):
        self.use_azure = os.getenv("AZURE_OPENAI_API_KEY") is not None
        
        if self.use_azure:
            self.client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
            self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4")
        else:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.default_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
        self.default_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion using OpenAI/Azure OpenAI"""
        try:
            temperature = temperature if temperature is not None else self.default_temperature
            max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
            
            if self.use_azure:
                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion from a simple prompt"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, **kwargs)
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # Split large batches to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if self.use_azure:
                    response = await self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch
                    )
                else:
                    response = await self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch
                    )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise Exception(f"Embedding API error: {str(e)}")
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]
    
    async def analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze text quality using OpenAI"""
        try:
            prompt = f"""
            Analyze the quality of this text content for research purposes:
            
            Text: {text[:1000]}...
            
            Provide analysis in the following format:
            {{
                "readability_score": 0.0-1.0,
                "information_density": 0.0-1.0,
                "credibility_indicators": ["list", "of", "indicators"],
                "potential_issues": ["list", "of", "issues"],
                "overall_quality": 0.0-1.0
            }}
            
            Return only valid JSON.
            """
            
            response = await self.generate_completion(prompt, temperature=0.1)
            
            # Try to parse JSON response
            import json
            try:
                quality_analysis = json.loads(response)
                return quality_analysis
            except json.JSONDecodeError:
                # Fallback to basic analysis
                return {
                    "readability_score": 0.7,
                    "information_density": 0.6,
                    "credibility_indicators": ["structured_content"],
                    "potential_issues": ["parsing_failed"],
                    "overall_quality": 0.6
                }
                
        except Exception as e:
            self.logger.error(f"Text quality analysis failed: {e}")
            return {
                "readability_score": 0.5,
                "information_density": 0.5,
                "credibility_indicators": [],
                "potential_issues": ["analysis_failed"],
                "overall_quality": 0.5
            }
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text"""
        try:
            prompt = f"""
            Summarize the following text in approximately {max_length} words, 
            focusing on the most important information:
            
            {text}
            
            Summary:
            """
            
            response = await self.generate_completion(
                prompt, 
                temperature=0.2,
                max_tokens=max_length + 50
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Text summarization failed: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text"""
        try:
            prompt = f"""
            Extract the {max_keywords} most important keywords/phrases from this text:
            
            {text[:2000]}
            
            Return only the keywords, one per line, no numbering or bullets.
            """
            
            response = await self.generate_completion(prompt, temperature=0.1)
            
            keywords = [line.strip() for line in response.split('\n') if line.strip()]
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "provider": "Azure OpenAI" if self.use_azure else "OpenAI",
            "model": self.model_name,
            "embedding_model": self.embedding_model,
            "deployment": self.deployment_name if self.use_azure else None,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens
        }
