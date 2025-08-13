import asyncio
from typing import List
from workflow.state import ResearchState, ResearchStep, Source
from services.vector_store import VectorStoreService
from services.openai_service import OpenAIService

class InternalResearchAgent:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.openai_service = OpenAIService()
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """Search and process internal documents"""
        try:
            questions = state["research_questions"]
            internal_sources = []
            
            # Process questions in parallel
            tasks = [self._research_question(question) for question in questions]
            question_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for i, results in enumerate(question_results):
                if isinstance(results, Exception):
                    continue
                    
                question = questions[i]
                for result in results:
                    source = Source(
                        content=result["text"],
                        title=result.get("title", f"Internal Document - {question}"),
                        url=None,
                        source_type="internal",
                        relevance_score=result["score"],
                        credibility_score=0.9,  # Internal sources are highly credible
                        question=question,
                        metadata=result.get("metadata", {})
                    )
                    internal_sources.append(source)
            
            # Remove duplicates and sort by relevance
            internal_sources = self._deduplicate_sources(internal_sources)
            internal_sources = sorted(internal_sources, 
                                    key=lambda x: x["relevance_score"], 
                                    reverse=True)[:20]  # Top 20 sources
            
            return {
                **state,
                "internal_sources": internal_sources,
                "current_step": ResearchStep.INTERNAL_RESEARCH if state["current_step"] != ResearchStep.EXTERNAL_RESEARCH else ResearchStep.RESEARCH_COMPLETE
            }
            
        except Exception as e:
            return self._handle_error(state, e)
    
    async def _research_question(self, question: str) -> List[dict]:
        """Research a single question in internal documents"""
        try:
            # Expand query with synonyms and related terms
            expanded_query = await self._expand_query(question)
            
            # Search with multiple strategies
            results = []
            
            # Primary search
            primary_results = await self.vector_store.similarity_search(
                query=question,
                k=10,
                threshold=0.6
            )
            results.extend(primary_results)
            
            # Expanded search
            if expanded_query != question:
                expanded_results = await self.vector_store.similarity_search(
                    query=expanded_query,
                    k=5,
                    threshold=0.5
                )
                results.extend(expanded_results)
            
            return results
            
        except Exception as e:
            print(f"Error researching question '{question}': {e}")
            return []
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        try:
            prompt = f"""
            Expand this research query with relevant synonyms and related terms: "{query}"
            
            Provide an expanded search query that captures the same intent but uses additional relevant terms.
            Keep it concise (max 20 words).
            
            Original: {query}
            Expanded:
            """
            
            response = await self.openai_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=50
            )
            
            return response.strip()
            
        except Exception:
            return query
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on content similarity"""
        unique_sources = []
        seen_content = set()
        
        for source in sources:
            # Create content hash
            content_hash = hash(source["content"][:200])  # First 200 chars
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_sources.append(source)
        
        return unique_sources
    
    def _handle_error(self, state: ResearchState, error: Exception) -> ResearchState:
        """Handle internal research errors"""
        error_info = {
            "type": "internal_research_failed",
            "message": str(error),
            "step": "internal_research"
        }
        
        errors = state.get("errors", [])
        errors.append(error_info)
        
        # Continue with empty internal sources
        return {
            **state,
            "internal_sources": [],
            "errors": errors,
            "current_step": ResearchStep.INTERNAL_RESEARCH if state["current_step"] != ResearchStep.EXTERNAL_RESEARCH else ResearchStep.RESEARCH_COMPLETE
        }
