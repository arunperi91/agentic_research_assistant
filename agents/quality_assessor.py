from typing import List, Dict, Any
from workflow.state import ResearchState, ResearchStep, Source, QualityAssessment
from services.openai_service import OpenAIService

class QualityAssessorAgent:
    def __init__(self):
        self.openai_service = OpenAIService()
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """Assess research quality and determine next steps"""
        try:
            internal_sources = state.get("internal_sources", [])
            external_sources = state.get("external_sources", [])
            questions = state["research_questions"]
            
            # Perform comprehensive quality assessment
            assessment = await self._assess_research_quality(
                questions, internal_sources, external_sources
            )
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(assessment)
            
            # Update state with assessment
            updated_state = {
                **state,
                "quality_assessment": assessment,
                "quality_score": overall_score,
                "all_sources": internal_sources + external_sources,
                "current_step": ResearchStep.QUALITY_ASSESSED
            }
            
            return updated_state
            
        except Exception as e:
            return self._handle_error(state, e)
    
    async def _assess_research_quality(
        self, 
        questions: List[str], 
        internal_sources: List[Source], 
        external_sources: List[Source]
    ) -> QualityAssessment:
        """Comprehensive quality assessment"""
        
        # 1. Source Diversity Assessment
        source_diversity = self._assess_source_diversity(internal_sources, external_sources)
        
        # 2. Content Coverage Assessment
        content_coverage = await self._assess_content_coverage(questions, internal_sources + external_sources)
        
        # 3. Source Credibility Assessment
        source_credibility = self._assess_source_credibility(internal_sources + external_sources)
        
        # 4. Information Gap Identification
        information_gaps = await self._identify_information_gaps(questions, internal_sources + external_sources)
        
        # 5. Calculate overall score
        overall_score = (source_diversity * 0.2 + 
                        content_coverage * 0.4 + 
                        source_credibility * 0.3 + 
                        (1.0 - len(information_gaps) / len(questions)) * 0.1)
        
        return QualityAssessment(
            source_diversity=source_diversity,
            content_coverage=content_coverage,
            source_credibility=source_credibility,
            information_gaps=information_gaps,
            overall_score=overall_score
        )
    
    def _assess_source_diversity(self, internal_sources: List[Source], external_sources: List[Source]) -> float:
        """Assess diversity of sources"""
        total_sources = len(internal_sources) + len(external_sources)
        
        if total_sources == 0:
            return 0.0
        
        # Check internal vs external balance
        internal_ratio = len(internal_sources) / total_sources
        external_ratio = len(external_sources) / total_sources
        
        # Ideal ratio is around 40% internal, 60% external
        ideal_internal = 0.4
        internal_score = 1.0 - abs(internal_ratio - ideal_internal) / ideal_internal
        
        # Check external domain diversity
        external_domains = set()
        for source in external_sources:
            domain = source.get("metadata", {}).get("domain", "unknown")
            external_domains.add(domain)
        
        domain_diversity = min(len(external_domains) / max(len(external_sources), 1), 1.0)
        
        # Combined diversity score
        diversity_score = (internal_score * 0.6 + domain_diversity * 0.4)
        
        return max(0.0, min(1.0, diversity_score))
    
    async def _assess_content_coverage(self, questions: List[str], sources: List[Source]) -> float:
        """Assess how well sources cover research questions"""
        if not questions or not sources:
            return 0.0
        
        coverage_scores = []
        
        for question in questions:
            # Find sources related to this question
            related_sources = [s for s in sources if s["question"] == question]
            
            if not related_sources:
                coverage_scores.append(0.0)
                continue
            
            # Calculate coverage for this question
            question_coverage = await self._calculate_question_coverage(question, related_sources)
            coverage_scores.append(question_coverage)
        
        return sum(coverage_scores) / len(coverage_scores)
    
    async def _calculate_question_coverage(self, question: str, sources: List[Source]) -> float:
        """Calculate how well sources answer a specific question"""
        if not sources:
            return 0.0
        
        try:
            # Use LLM to assess coverage
            source_texts = "\n\n".join([f"Source: {s['content'][:500]}..." for s in sources[:5]])
            
            prompt = f"""
            Question: {question}
            
            Available sources:
            {source_texts}
            
            Rate how well these sources answer the question on a scale of 0.0 to 1.0:
            - 1.0: Question is completely and comprehensively answered
            - 0.8: Question is well answered with minor gaps
            - 0.6: Question is partially answered
            - 0.4: Question is minimally addressed
            - 0.2: Question is barely touched upon
            - 0.0: Question is not addressed at all
            
            Provide only a single number between 0.0 and 1.0:
            """
            
            response = await self.openai_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            try:
                score = float(response.strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                return 0.5
                
        except Exception:
            # Fallback: simple relevance-based scoring
            avg_relevance = sum([s["relevance_score"] for s in sources]) / len(sources)
            return avg_relevance
    
    def _assess_source_credibility(self, sources: List[Source]) -> float:
        """Assess overall credibility of sources"""
        if not sources:
            return 0.0
        
        credibility_scores = [s.get("credibility_score", 0.5) for s in sources]
        return sum(credibility_scores) / len(credibility_scores)
    
    async def _identify_information_gaps(self, questions: List[str], sources: List[Source]) -> List[str]:
        """Identify questions with insufficient information"""
        gaps = []
        
        for question in questions:
            related_sources = [s for s in sources if s["question"] == question]
            
            if not related_sources:
                gaps.append(question)
                continue
            
            # Check if sources provide substantial information
            total_content = sum([len(s["content"]) for s in related_sources])
            avg_relevance = sum([s["relevance_score"] for s in related_sources]) / len(related_sources)
            
            # Thresholds for sufficient information
            if total_content < 1000 or avg_relevance < 0.4:
                gaps.append(question)
        
        return gaps
    
    def _calculate_overall_score(self, assessment: QualityAssessment) -> float:
        """Calculate weighted overall quality score"""
        return assessment["overall_score"]
    
    def _handle_error(self, state: ResearchState, error: Exception) -> ResearchState:
        """Handle quality assessment errors"""
        error_info = {
            "type": "quality_assessment_failed",
            "message": str(error),
            "step": "quality_assessment"
        }
        
        errors = state.get("errors", [])
        errors.append(error_info)
        
        # Default to medium quality score
        return {
            **state,
            "quality_score": 0.6,
            "all_sources": state.get("internal_sources", []) + state.get("external_sources", []),
            "errors": errors,
            "current_step": ResearchStep.QUALITY_ASSESSED
        }
