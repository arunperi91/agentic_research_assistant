import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from workflow.research_workflow import create_research_workflow
from workflow.state import ResearchState, ResearchStep
from utils.document_utils import create_professional_word_doc
from services.vector_store import VectorStoreService

app = FastAPI(title="Agentic Research Assistant", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store = VectorStoreService()
research_workflow = create_research_workflow()

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup if needed"""
    try:
        if not vector_store.is_data_loaded():
            print("🔄 No data found in vector store. Initializing from data folder...")
            result = await vector_store.initialize_from_data_folder()
            
            if result["status"] == "completed":
                print(f"✅ Initialized with {result['successful']} files")
            elif result["status"] == "empty":
                print("⚠️ No PDF files found in data folder")
            else:
                print(f"❌ Failed to initialize data: {result.get('message', 'Unknown error')}")
        else:
            print("✅ Data already loaded in vector store")
    except Exception as e:
        print(f"❌ Startup initialization failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "Agentic Research Assistant API", 
        "version": "2.0.0",
        "data_status": vector_store.get_data_folder_status()
    }

@app.get("/data/status")
async def get_data_status():
    """Get current data folder and vector store status"""
    return vector_store.get_data_folder_status()

@app.post("/data/reload")
async def reload_data():
    """Reload data from data folder"""
    try:
        result = await vector_store.initialize_from_data_folder()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload data: {str(e)}")

@app.post("/research/")
async def conduct_research(request: Request):
    """Main research endpoint using agentic workflow"""
    try:
        data = await request.json()
        topic = data.get("topic")
        
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        # Check if data is loaded
        if not vector_store.is_data_loaded():
            raise HTTPException(
                status_code=400, 
                detail="No data loaded. Please add PDF files to data folder and restart the application."
            )
        
        # Initialize research state
        initial_state = ResearchState(
            topic=topic,
            research_plan=None,
            research_questions=[],
            internal_sources=[],
            external_sources=[],
            all_sources=[],
            quality_assessment=None,
            quality_score=0.0,
            report_sections={},
            final_report=None,
            current_step=ResearchStep.INITIALIZED,
            iteration_count=0,
            errors=[],
            start_time=datetime.now().isoformat(),
            end_time=None,
            processing_time=None
        )
        
        # Execute workflow
        final_state = await research_workflow.execute(initial_state)
        
        # Create Word document
        if final_state.get("final_report"):
            doc_path = create_professional_word_doc(
                final_state["final_report"],
                topic,
                final_state
            )
            
            # Return file response
            filename = f"{topic.replace(' ', '_')}_research_report.docx"
            return FileResponse(
                doc_path,
                filename=filename,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate report")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

@app.post("/plan/")
async def generate_plan(request: Request):
    """Generate research plan only"""
    try:
        data = await request.json()
        topic = data.get("topic")
        
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        # Check if data is loaded
        if not vector_store.is_data_loaded():
            raise HTTPException(
                status_code=400, 
                detail="No data loaded. Please add PDF files to data folder."
            )
        
        # Initialize minimal state for planning
        initial_state = ResearchState(
            topic=topic,
            research_plan=None,
            research_questions=[],
            internal_sources=[],
            external_sources=[],
            all_sources=[],
            quality_assessment=None,
            quality_score=0.0,
            report_sections={},
            final_report=None,
            current_step=ResearchStep.INITIALIZED,
            iteration_count=0,
            errors=[],
            start_time=datetime.now().isoformat(),
            end_time=None,
            processing_time=None
        )
        
        # Execute only planning
        planner = research_workflow.planner
        planned_state = await planner.execute(initial_state)
        
        return {
            "topic": topic,
            "research_plan": planned_state.get("research_plan"),
            "research_questions": planned_state.get("research_questions", []),
            "status": "plan_generated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
