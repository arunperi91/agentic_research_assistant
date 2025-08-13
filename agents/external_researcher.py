import asyncio
from typing import List, Dict, Any
from workflow.state import ResearchState, ResearchStep, Source
from services.web_search_service import WebSearchService
from services.openai_service import OpenAIService

class ExternalResearchAgent:
    def __init__(self):
        self.web_search = WebSearchService()
        self.openai_service = OpenAIService()
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """Conduct comprehensive web research"""
        try:
            questions = state["research_questions"]
            external_sources = []
            
            # Process questions in parallel with rate limiting
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent searches
            tasks = [self._research_question_with_semaphore(semaphore, question) 
                    for question in questions]
            
            question_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect and process results
            for i, results in enumerate(question_results):
                if isinstance(results, Exception):
                    continue
                
                question = questions[i]
                for result in results:
                    source = Source(
                        content=result["content"],
                        title=result["title"],
                        url=result["url"],
                        source_type="external",
                        relevance_score=result.get("relevance_score", 0.5),
                        credibility_score=result.get("credibility_score", 0.6),
                        question=question,
                        metadata=result.get("metadata", {})
                    )
                    external_sources.append(source)
            
            # Filter and rank sources
            external_sources = await self._filter_quality_sources(external_sources)
            external_sources = sorted(external_sources, 
                                    key=lambda x: (x["credibility_score"] * x["relevance_score"]), 
                                    reverse=True)[:25]  # Top 25 sources
            
            return {
                **state,
                "external_sources": external_sources,
                "current_step": ResearchStep.EXTERNAL_RESEARCH if state["current_step"] != ResearchStep.INTERNAL_RESEARCH else ResearchStep.RESEARCH_COMPLETE
            }
            
        except Exception as e:
            return self._handle_error(state, e)
    
    async def _research_question_with_semaphore(self, semaphore: asyncio.Semaphore, question: str) -> List[Dict]:
        """Research with rate limiting"""
        async with semaphore:
            return await self._research_question(question)
    
    async def _research_question(self, question: str) -> List[Dict]:
        """Research a single question using multiple search strategies"""
        try:
            all_results = []
            
            # Strategy 1: Direct question search
            direct_results = await self.web_search.search(question, max_results=5)
            all_results.extend(direct_results)
            
            # Strategy 2: Keyword-based search
            keywords = await self._extract_keywords(question)
            if keywords:
                keyword_results = await self.web_search.search(keywords, max_results=3)
                all_results.extend(keyword_results)
            
            # Strategy 3: Alternative phrasing
            alt_question = await self._rephrase_question(question)
            if alt_question != question:
                alt_results = await self.web_search.search(alt_question, max_results=3)
                all_results.extend(alt_results)
            
            # Process and score results
            processed_results = []
            for result in all_results:
                processed_result = await self._process_search_result(result, question)
                if processed_result:
                    processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            print(f"Error researching question '{question}': {e}")
            return []
    
    async def _extract_keywords(self, question: str) -> str:
        """Extract key search terms from question"""
        try:
            prompt = f"""
            Extract the most important keywords for web search from this research question: "{question}"
            
            Return only the keywords separated by spaces, no extra text.
            Focus on nouns, technical terms, and specific concepts.
            Maximum 6 keywords.
            """
            
            response = await self.openai_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=30
            )
            
            return response.strip()
            
        except Exception:
            return ""
    
    async def _rephrase_question(self, question: str) -> str:
        """Generate alternative phrasing for better search results"""
        try:
            prompt = f"""
            Rephrase this research question for better web search results: "{question}"
            
            Make it more specific and search-friendly while keeping the same meaning.
            Use different words but maintain the intent.
            
            Original: {question}
            Rephrased:
            """
            
            response = await self.openai_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            return response.strip()
            
        except Exception:
            return question
    
    async def _process_search_result(self, result: Dict, question: str) -> Dict:
        """Process and score search result"""
        try:
            # Extract content if needed
            if "content" not in result or not result["content"]:
                result["content"] = await self.web_search.extract_content(result["url"])
            
            # Score relevance
            relevance_score = await self._score_relevance(result["content"], question)
            
            # Score credibility based on domain and content
            credibility_score = self._score_credibility(result)
            
            return {
                "content": result["content"],
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "relevance_score": relevance_score,
                "credibility_score": credibility_score,
                "metadata": {
                    "domain": result.get("domain", ""),
                    "published_date": result.get("published_date", ""),
                    "word_count": len(result["content"].split())
                }
            }
            
        except Exception as e:
            print(f"Error processing search result: {e}")
            return None
    
    async def _score_relevance(self, content: str, question: str) -> float:
        """Score content relevance to question"""
        try:
            # Simple keyword-based scoring for now
            question_words = set(question.lower().split())
            content_words = set(content.lower().split())
            
            overlap = len(question_words.intersection(content_words))
            relevance = min(overlap / len(question_words), 1.0)
            
            return max(relevance, 0.1)  # Minimum score
            
        except Exception:
            return 0.5
    
    def _score_credibility(self, result: Dict) -> float:
        """Score source credibility"""
        credibility = 0.5  # Base score
        
        domain = result.get("domain", "").lower()
        url = result.get("url", "").lower()
        
        # High credibility domains
        if any(trusted in domain for trusted in [
            "edu", "gov", "org", "ieee", "acm", "springer", "nature", "science"
        ]):
            credibility += 0.3
        
        # Medium credibility domains
        elif any(med in domain for med in [
            "wikipedia", "stackoverflow", "medium", "github"
        ]):
            credibility += 0.1
        
        # Check for HTTPS
        if url.startswith("https"):
            credibility += 0.1
        
        # Content length (longer articles often more credible)
        content_length = len(result.get("content", ""))
        if content_length > 1000:
            credibility += 0.1
        
        return min(credibility, 1.0)
    
    async def _filter_quality_sources(self, sources: List[Source]) -> List[Source]:
        """Filter sources based on quality metrics"""
        quality_sources = []
        
        for source in sources:
            # Minimum thresholds
            if (source["relevance_score"] >= 0.3 and 
                source["credibility_score"] >= 0.4 and
                len(source["content"]) >= 200):  # Minimum content length
                
                quality_sources.append(source)
        
        return quality_sources
    
    def _handle_error(self, state: ResearchState, error: Exception) -> ResearchState:
        """Handle external research errors"""
        error_info = {
            "type": "external_research_failed",
            "message": str(error),
            "step": "external_research"
        }
        
        errors = state.get("errors", [])
        errors.append(error_info)
        
        # Continue with empty external sources
        return {
            **state,
            "external_sources": [],
            "errors": errors,
            "current_step": ResearchStep.EXTERNAL_RESEARCH if state["current_step"] != ResearchStep.INTERNAL_RESEARCH else ResearchStep.RESEARCH_COMPLETE
        }
