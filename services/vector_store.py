import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import os
import asyncio
from services.openai_service import OpenAIService
import logging
import uuid
from datetime import datetime
import glob
import warnings

# Suppress ONNX warnings
warnings.filterwarnings("ignore", message=".*ONNX Runtime.*")

class VectorStoreService:
    def __init__(self):
        # Initialize logger FIRST - before any other operations
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.openai_service = OpenAIService()
            
            # ChromaDB configuration
            self.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_persist")
            self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "research_documents")
            self.data_folder = os.getenv("DATA_FOLDER_PATH", "./data/documents")
            
            # Ensure directories exist
            os.makedirs(self.persist_directory, exist_ok=True)
            os.makedirs(self.data_folder, exist_ok=True)
            
            # Initialize ChromaDB client with better error handling
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self._get_or_create_collection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorStoreService: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            self.logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except ValueError:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research documents collection"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
            return collection
        except Exception as e:
            self.logger.error(f"Failed to get or create collection: {e}")
            raise
    
    async def initialize_from_data_folder(self) -> Dict[str, Any]:
        """Initialize vector store with all files from data folder"""
        try:
            if not os.path.exists(self.data_folder):
                self.logger.warning(f"Data folder not found: {self.data_folder}")
                return {"status": "error", "message": "Data folder not found"}
            
            # Find all PDF files in data folder
            pdf_files = glob.glob(os.path.join(self.data_folder, "**", "*.pdf"), recursive=True)
            
            if not pdf_files:
                self.logger.info("No PDF files found in data folder")
                return {"status": "empty", "message": "No PDF files found", "files_processed": 0}
            
            self.logger.info(f"Found {len(pdf_files)} PDF files in data folder")
            
            # Import PDF service here to avoid circular imports
            from services.pdf_service import PDFService
            pdf_service = PDFService()
            
            # Process all files
            results = []
            successful_count = 0
            
            for pdf_file in pdf_files:
                try:
                    filename = os.path.basename(pdf_file)
                    self.logger.info(f"Processing: {filename}")
                    
                    result = await pdf_service.process_pdf(pdf_file, filename)
                    results.append(result)
                    
                    if result["status"] == "success":
                        successful_count += 1
                        self.logger.info(f"✅ Processed {filename}: {result.get('chunks', 0)} chunks")
                    elif result["status"] == "already_exists":
                        successful_count += 1
                        self.logger.info(f"📋 Already processed: {filename}")
                    else:
                        self.logger.error(f"❌ Failed to process {filename}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    self.logger.error(f"Exception processing {pdf_file}: {e}")
                    results.append({"status": "error", "filename": os.path.basename(pdf_file), "error": str(e)})
            
            return {
                "status": "completed",
                "total_files": len(pdf_files),
                "successful": successful_count,
                "failed": len(pdf_files) - successful_count,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize from data folder: {e}")
            return {"status": "error", "message": str(e)}
    
    def is_data_loaded(self) -> bool:
        """Check if data folder files have been loaded"""
        try:
            count = self.collection.count()
            return count > 0
        except Exception:
            return False
    
    def get_data_folder_status(self) -> Dict[str, Any]:
        """Get status of data folder files"""
        try:
            pdf_files = glob.glob(os.path.join(self.data_folder, "**", "*.pdf"), recursive=True)
            collection_count = self.collection.count()
            
            return {
                "data_folder": self.data_folder,
                "pdf_files_found": len(pdf_files),
                "documents_in_collection": collection_count,
                "is_loaded": collection_count > 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ... (rest of the existing methods remain the same)
    
    async def add_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to the vector store"""
        try:
            if not documents:
                return False
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = await self.openai_service.generate_embeddings(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 10, 
        threshold: float = 0.0,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = await self.openai_service.generate_single_embedding(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    if similarity_score >= threshold:
                        result = {
                            "text": doc,
                            "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                            "score": similarity_score,
                            "distance": distance
                        }
                        processed_results.append(result)
            
            self.logger.info(f"Found {len(processed_results)} documents above threshold {threshold}")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    # ... (include all other existing methods from the previous version)
