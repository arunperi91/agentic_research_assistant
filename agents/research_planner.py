import asyncio
from typing import List, Dict, Any
from workflow.state import ResearchState, ResearchStep
from services.openai_service import OpenAIService
from services.vector_store import VectorStoreService
from services.web_search_service import WebSearchService

class ResearchPlannerAgent:
    def __init__(self):
        self.openai_service = OpenAIService()
        self.vector_store = VectorStoreService()
        self.web_search = WebSearchService()
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """Generate comprehensive research plan"""
        try:
            topic = state["topic"]
            
            # Generate research questions
            research_questions = await self._generate_research_questions(topic)
            
            # Discover available sources
            internal_preview = await self._discover_internal_sources(topic)
            external_preview = await self._preview_external_sources(topic)
            
            # Create research plan
            plan = {
                "topic": topic,
                "questions": research_questions,
                "internal_sources_available": len(internal_preview),
                "external_domains_identified": len(external_preview),
                "estimated_duration_minutes": self._estimate_duration(research_questions),
                "research_strategy": "comprehensive_multi_source"
            }
            
            return {
                **state,
                "research_plan": plan,
                "research_questions": research_questions,
                "current_step": ResearchStep.PLANNING_COMPLETE,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            return self._handle_error(state, e, "planning_failed")
    
    async def _generate_research_questions(self, topic: str) -> List[str]:
        """Generate focused research questions"""
        prompt = f"""
        Generate 4-6 specific, focused research questions for the topic: "{topic}"
        
        Requirements:
        - Questions should be specific and answerable through research
        - Cover different aspects of the topic (technical, business, social impact, etc.)
        - Avoid overly broad or philosophical questions
        - Each question should lead to actionable research
        
        Format as a numbered list.
        """
        
        response = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Parse questions from response
        questions = []
        for line in response.strip().split('\n'):
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                question = line.split('.', 1)[-1].strip()
                if question.endswith('?'):
                    questions.append(question)
        
        return questions[:6]  # Limit to 6 questions
    
    async def _discover_internal_sources(self, topic: str) -> List[Dict]:
        """Preview available internal sources"""
        try:
            results = await self.vector_store.similarity_search(
                query=topic,
                k=10,
                threshold=0.3
            )
            return results
        except Exception:
            return []
    
    async def _preview_external_sources(self, topic: str) -> List[Dict]:
        """Preview external source domains"""
        try:
            search_results = await self.web_search.preview_search(topic, max_results=5)
            domains = list(set([result.get('domain', 'unknown') for result in search_results]))
            return domains
        except Exception:
            return []
    
    def _estimate_duration(self, questions: List[str]) -> int:
        """Estimate research duration in minutes"""
        base_time = 5  # minutes per question
        complexity_multiplier = 1.5 if len(questions) > 4 else 1.2
        return int(len(questions) * base_time * complexity_multiplier)
    
    def _handle_error(self, state: ResearchState, error: Exception, error_type: str) -> ResearchState:
        """Handle planning errors"""
        error_info = {
            "type": error_type,
            "message": str(error),
            "step": "planning"
        }
        
        errors = state.get("errors", [])
        errors.append(error_info)
        
        return {
            **state,
            "errors": errors,
            "current_step": ResearchStep.ERROR_RECOVERY
        }
