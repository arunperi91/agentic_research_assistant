import asyncio
from typing import Dict, List
from workflow.state import ResearchState, ResearchStep, Source
from services.openai_service import OpenAIService

class ReportGeneratorAgent:
    def __init__(self):
        self.openai_service = OpenAIService()
    
    async def execute(self, state: ResearchState) -> ResearchState:
        """Generate structured report sections"""
        try:
            topic = state["topic"]
            sources = state.get("all_sources", [])
            questions = state["research_questions"]
            
            if not sources:
                return self._handle_insufficient_sources(state)
            
            # Generate sections in parallel
            section_tasks = {
                "executive_summary": self._generate_executive_summary(topic, sources, questions),
                "introduction": self._generate_introduction(topic, sources),
                "key_findings": self._generate_key_findings(topic, sources, questions),
                "detailed_analysis": self._generate_detailed_analysis(topic, sources, questions),
                "conclusion": self._generate_conclusion(topic, sources),
                "references": self._generate_references(sources)
            }
            
            # Execute all tasks
            section_results = await asyncio.gather(
                *section_tasks.values(),
                return_exceptions=True
            )
            
            # Collect results
            sections = {}
            for i, (section_name, result) in enumerate(zip(section_tasks.keys(), section_results)):
                if isinstance(result, Exception):
                    sections[section_name] = f"Error generating {section_name}: {str(result)}"
                else:
                    sections[section_name] = result
            
            return {
                **state,
                "report_sections": sections,
                "current_step": ResearchStep.SECTIONS_GENERATED
            }
            
        except Exception as e:
            return self._handle_error(state, e)
    
    async def _generate_executive_summary(self, topic: str, sources: List[Source], questions: List[str]) -> str:
        """Generate executive summary"""
        # Prepare key points from sources
        key_points = self._extract_key_points(sources[:10])  # Top 10 sources
        
        prompt = f"""
        Create a comprehensive executive summary for a research report on: "{topic}"
        
        Research Questions Addressed:
        {chr(10).join([f"• {q}" for q in questions])}
        
        Key Information Available:
        {chr(10).join(key_points)}
        
        Requirements:
        - 250-400 words
        - Highlight the most important findings
        - Provide clear, actionable insights
        - Maintain professional tone
        - Include quantitative data where available
        
        Executive Summary:
        """
        
        response = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.strip()
    
    async def _generate_introduction(self, topic: str, sources: List[Source]) -> str:
        """Generate introduction section"""
        context_info = self._extract_context(sources[:5])
        
        prompt = f"""
        Write a comprehensive introduction for a research report on: "{topic}"
        
        Context Information:
        {chr(10).join(context_info)}
        
        Requirements:
        - 200-350 words
        - Define the topic and its significance
        - Provide background context
        - Outline the scope of research
        - Set expectations for the reader
        
        Introduction:
        """
        
        response = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=450
        )
        
        return response.strip()
    
    async def _generate_key_findings(self, topic: str, sources: List[Source], questions: List[str]) -> str:
        """Generate key findings section"""
        findings_data = self._organize_findings_by_question(sources, questions)
        
        prompt = f"""
        Generate a "Key Findings" section for a research report on: "{topic}"
        
        Organize findings by research question:
        
        {findings_data}
        
        Requirements:
        - Use bullet points and subheadings
        - Highlight the most significant discoveries
        - Include specific data and examples where available
        - Ensure findings are supported by sources
        - 400-600 words
        
        Key Findings:
        """
        
        response = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700
        )
        
        return response.strip()
    
    async def _generate_detailed_analysis(self, topic: str, sources: List[Source], questions: List[str]) -> str:
        """Generate detailed analysis section"""
        analysis_content = self._prepare_analysis_content(sources, questions)
        
        prompt = f"""
        Create a detailed analysis section for a research report on: "{topic}"
        
        Analysis Content:
        {analysis_content}
        
        Requirements:
        - Provide in-depth analysis of findings
        - Draw connections between different sources
        - Identify patterns and trends
        - Discuss implications and significance
        - Use subheadings to organize content
        - 600-800 words
        - Cite sources appropriately
        
        Detailed Analysis:
        """
        
        response = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=900
        )
        
        return response.strip()
    
    async def _generate_conclusion(self, topic: str, sources: List[Source]) -> str:
        """Generate conclusion section"""
        conclusion_points = self._extract_conclusion_points(sources)
        
        prompt = f"""
        Write a comprehensive conclusion for a research report on: "{topic}"
        
        Key Points to Address:
        {chr(10).join(conclusion_points)}
        
        Requirements:
        - Summarize main findings
        - Discuss implications and significance
        - Identify limitations of the research
        - Suggest areas for future research
        - Provide actionable recommendations if applicable
        - 300-450 words
        
        Conclusion:
        """
        
        response = await self.openai_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=550
        )
        
        return response.strip()
    
    async def _generate_references(self, sources: List[Source]) -> str:
        """Generate references section"""
        references = []
        
        # Group sources by type
        internal_sources = [s for s in sources if s["source_type"] == "internal"]
        external_sources = [s for s in sources if s["source_type"] == "external"]
        
        # Internal sources
        if internal_sources:
            references.append("## Internal Sources")
            for i, source in enumerate(internal_sources[:15], 1):
                title = source.get("title", "Internal Document")
                references.append(f"{i}. {title}")
        
        # External sources
        if external_sources:
            references.append("\n## External Sources")
            for i, source in enumerate(external_sources[:20], 1):
                title = source.get("title", "Web Source")
                url = source.get("url", "")
                if url:
                    references.append(f"{i}. {title}. Retrieved from {url}")
                else:
                    references.append(f"{i}. {title}")
        
        return "\n".join(references)
    
    def _extract_key_points(self, sources: List[Source]) -> List[str]:
        """Extract key points from sources"""
        key_points = []
        for source in sources[:10]:  # Limit to top sources
            content_preview = source["content"][:300]
            key_points.append(f"• {content_preview}...")
        return key_points
    
    def _extract_context(self, sources: List[Source]) -> List[str]:
        """Extract context information"""
        context = []
        for source in sources:
            content_preview = source["content"][:200]
            context.append(f"• {content_preview}...")
        return context
    
    def _organize_findings_by_question(self, sources: List[Source], questions: List[str]) -> str:
        """Organize findings by research question"""
        organized_content = []
        
        for question in questions:
            related_sources = [s for s in sources if s["question"] == question]
            if related_sources:
                organized_content.append(f"\n**{question}**")
                for source in related_sources[:3]:  # Top 3 sources per question
                    content_preview = source["content"][:200]
                    organized_content.append(f"- {content_preview}...")
        
        return "\n".join(organized_content)
    
    def _prepare_analysis_content(self, sources: List[Source], questions: List[str]) -> str:
        """Prepare content for detailed analysis"""
        analysis_content = []
        
        # High-quality sources for analysis
        quality_sources = sorted(sources, 
                                key=lambda x: x["relevance_score"] * x.get("credibility_score", 0.5), 
                                reverse=True)[:15]
        
        for source in quality_sources:
            content_excerpt = source["content"][:400]
            analysis_content.append(f"Source: {source.get('title', 'Untitled')}")
            analysis_content.append(f"Content: {content_excerpt}...")
            analysis_content.append("")
        
        return "\n".join(analysis_content)
    
    def _extract_conclusion_points(self, sources: List[Source]) -> List[str]:
        """Extract key points for conclusion"""
        conclusion_points = []
        
        # Get diverse sources for conclusion
        internal_sources = [s for s in sources if s["source_type"] == "internal"][:3]
        external_sources = [s for s in sources if s["source_type"] == "external"][:5]
        
        all_conclusion_sources = internal_sources + external_sources
        
        for source in all_conclusion_sources:
            content_preview = source["content"][:250]
            conclusion_points.append(f"• {content_preview}...")
        
        return conclusion_points
    
    def _handle_insufficient_sources(self, state: ResearchState) -> ResearchState:
        """Handle case with insufficient sources"""
        topic = state["topic"]
        
        minimal_sections = {
            "executive_summary": f"Research on '{topic}' was conducted but insufficient sources were found to generate a comprehensive report.",
            "introduction": f"This report examines '{topic}'. However, limited source material was available for analysis.",
            "key_findings": "Limited findings available due to insufficient source material.",
            "detailed_analysis": "Unable to provide detailed analysis due to lack of sufficient sources.",
            "conclusion": "This research was limited by insufficient source material. Further investigation with additional resources is recommended.",
            "references": "No sufficient sources available for citation."
        }
        
        return {
            **state,
            "report_sections": minimal_sections,
            "current_step": ResearchStep.SECTIONS_GENERATED
        }
    
    def _handle_error(self, state: ResearchState, error: Exception) -> ResearchState:
        """Handle report generation errors"""
        error_info = {
            "type": "report_generation_failed",
            "message": str(error),
            "step": "report_generation"
        }
        
        errors = state.get("errors", [])
        errors.append(error_info)
        
        return {
            **state,
            "report_sections": {},
            "errors": errors,
            "current_step": ResearchStep.ERROR_RECOVERY
        }
