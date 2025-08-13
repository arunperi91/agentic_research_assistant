import asyncio
from datetime import datetime
from langgraph.graph import StateGraph, END
from workflow.state import ResearchState, ResearchStep
from agents.research_planner import ResearchPlannerAgent
from agents.internal_researcher import InternalResearchAgent
from agents.external_researcher import ExternalResearchAgent
from agents.quality_assessor import QualityAssessorAgent
from agents.report_generator import ReportGeneratorAgent
from agents.report_synthesizer import ReportSynthesizerAgent

class ResearchWorkflow:
    def __init__(self):
        self.planner = ResearchPlannerAgent()
        self.internal_researcher = InternalResearchAgent()
        self.external_researcher = ExternalResearchAgent()
        self.quality_assessor = QualityAssessorAgent()
        self.report_generator = ReportGeneratorAgent()
        self.synthesizer = ReportSynthesizerAgent()
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add agent nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("parallel_research", self._parallel_research_node)
        workflow.add_node("quality_assessor", self._quality_assessment_node)
        workflow.add_node("report_generator", self._report_generation_node)
        workflow.add_node("synthesizer", self._synthesis_node)
        workflow.add_node("error_recovery", self._error_recovery_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Define workflow edges
        workflow.add_edge("planner", "parallel_research")
        workflow.add_edge("parallel_research", "quality_assessor")
        
        # Conditional routing after quality assessment
        workflow.add_conditional_edges(
            "quality_assessor",
            self._quality_router,
            {
                "generate_report": "report_generator",
                "retry_research": "parallel_research",
                "replan": "planner",
                "error": "error_recovery"
            }
        )
        
        workflow.add_edge("report_generator", "synthesizer")
        workflow.add_edge("synthesizer", END)
        workflow.add_edge("error_recovery", END)
        
        return workflow.compile()
    
    async def _planning_node(self, state: ResearchState) -> ResearchState:
        """Execute research planning"""
        state["current_step"] = ResearchStep.PLANNING
        if not state.get("start_time"):
            state["start_time"] = datetime.now().isoformat()
        
        return await self.planner.execute(state)
    
    async def _parallel_research_node(self, state: ResearchState) -> ResearchState:
        """Execute parallel internal and external research"""
        state["current_step"] = ResearchStep.INTERNAL_RESEARCH
        
        # Run internal and external research in parallel
        internal_task = asyncio.create_task(self.internal_researcher.execute(state))
        external_task = asyncio.create_task(self.external_researcher.execute(state))
        
        # Wait for both to complete
        internal_result, external_result = await asyncio.gather(
            internal_task, external_task, return_exceptions=True
        )
        
        # Merge results
        merged_state = state.copy()
        
        if not isinstance(internal_result, Exception):
            merged_state["internal_sources"] = internal_result.get("internal_sources", [])
        else:
            merged_state["internal_sources"] = []
            errors = merged_state.get("errors", [])
            errors.append({
                "type": "internal_research_failed",
                "message": str(internal_result),
                "step": "internal_research"
            })
            merged_state["errors"] = errors
        
        if not isinstance(external_result, Exception):
            merged_state["external_sources"] = external_result.get("external_sources", [])
        else:
            merged_state["external_sources"] = []
            errors = merged_state.get("errors", [])
            errors.append({
                "type": "external_research_failed", 
                "message": str(external_result),
                "step": "external_research"
            })
            merged_state["errors"] = errors
        
        merged_state["current_step"] = ResearchStep.RESEARCH_COMPLETE
        return merged_state
    
    async def _quality_assessment_node(self, state: ResearchState) -> ResearchState:
        """Execute quality assessment"""
        state["current_step"] = ResearchStep.QUALITY_ASSESSMENT
        return await self.quality_assessor.execute(state)
    
    async def _report_generation_node(self, state: ResearchState) -> ResearchState:
        """Execute report generation"""
        state["current_step"] = ResearchStep.REPORT_GENERATION
        return await self.report_generator.execute(state)
    
    async def _synthesis_node(self, state: ResearchState) -> ResearchState:
        """Execute final report synthesis"""
        state["current_step"] = ResearchStep.REPORT_SYNTHESIS
        return await self.synthesizer.execute(state)
    
    async def _error_recovery_node(self, state: ResearchState) -> ResearchState:
        """Handle error recovery"""
        errors = state.get("errors", [])
        
        # Basic error recovery - could be expanded
        if len(errors) > 5:
            # Too many errors, fail gracefully
            state["final_report"] = f"Research failed due to multiple errors: {[e['type'] for e in errors]}"
            state["current_step"] = ResearchStep.REPORT_COMPLETE
        
        return state
    
    def _quality_router(self, state: ResearchState) -> str:
        """Route based on quality assessment"""
        quality_score = state.get("quality_score", 0.0)
        iteration_count = state.get("iteration_count", 0)
        errors = state.get("errors", [])
        
        # Check for critical errors
        critical_errors = [e for e in errors if "critical" in e.get("type", "")]
        if critical_errors:
            return "error"
        
        # Check iteration limit
        if iteration_count >= 3:
            return "generate_report"  # Max iterations reached
        
        # Quality-based routing
        if quality_score >= 0.7:
            return "generate_report"  # Good quality
        elif quality_score >= 0.4:
            return "retry_research"  # Moderate quality, try more research
        else:
            return "replan"  # Poor quality, replan research
    
    async def execute(self, initial_state: ResearchState) -> ResearchState:
        """Execute the complete research workflow"""
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            return final_state
        except Exception as e:
            # Fallback error handling
            error_state = {
                **initial_state,
                "final_report": f"Workflow execution failed: {str(e)}",
                "errors": initial_state.get("errors", []) + [{
                    "type": "workflow_execution_failed",
                    "message": str(e),
                    "step": "workflow_execution"
                }],
                "current_step": ResearchStep.REPORT_COMPLETE,
                "end_time": datetime.now().isoformat()
            }
            return error_state

# Create workflow instance
def create_research_workflow() -> ResearchWorkflow:
    """Factory function to create research workflow"""
    return ResearchWorkflow()
