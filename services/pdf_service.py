import PyPDF2
import os
import tempfile
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from services.vector_store import VectorStoreService

class PDFService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_file_size = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50")) * 1024 * 1024  # Convert to bytes
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            # Check file size
            if os.path.getsize(file_path) > self.max_file_size:
                raise Exception(f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB")
            
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                            text += "\n"
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        try:
            chunks = []
            
            # Simple chunking strategy
            words = text.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                
                if current_length + word_length > self.chunk_size and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": len(chunks),
                        "chunk_length": len(chunk_text),
                        "word_count": len(current_chunk)
                    })
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            # Add final chunk if it exists
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "chunk_length": len(chunk_text),
                    "word_count": len(current_chunk)
                })
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            self.logger.info(f"Created {len(chunks)} chunks from text")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Text chunking failed: {e}")
            return []
    
    def extract_metadata_from_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from PDF file"""
        try:
            metadata = {
                "source": filename,
                "file_type": "pdf",
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "processed_at": datetime.now().isoformat(),
                "file_hash": self._calculate_file_hash(file_path)
            }
            
            # Try to extract PDF metadata
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Basic PDF info
                    metadata.update({
                        "page_count": len(pdf_reader.pages),
                        "is_encrypted": pdf_reader.is_encrypted
                    })
                    
                    # PDF metadata
                    if pdf_reader.metadata:
                        pdf_meta = pdf_reader.metadata
                        metadata.update({
                            "title": pdf_meta.get("/Title", "").strip() if pdf_meta.get("/Title") else filename,
                            "author": pdf_meta.get("/Author", "").strip() if pdf_meta.get("/Author") else "Unknown",
                            "subject": pdf_meta.get("/Subject", "").strip() if pdf_meta.get("/Subject") else "",
                            "creator": pdf_meta.get("/Creator", "").strip() if pdf_meta.get("/Creator") else "",
                            "producer": pdf_meta.get("/Producer", "").strip() if pdf_meta.get("/Producer") else "",
                            "creation_date": str(pdf_meta.get("/CreationDate", "")) if pdf_meta.get("/CreationDate") else "",
                            "modification_date": str(pdf_meta.get("/ModDate", "")) if pdf_meta.get("/ModDate") else ""
                        })
                    else:
                        metadata.update({
                            "title": filename,
                            "author": "Unknown"
                        })
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract PDF metadata: {e}")
                metadata.update({
                    "title": filename,
                    "author": "Unknown",
                    "page_count": 0
                })
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return {
                "source": filename,
                "file_type": "pdf",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for duplicate detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"
    
    async def process_pdf(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Process PDF file: extract text, chunk, and add to vector store"""
        try:
            if filename is None:
                filename = os.path.basename(file_path)
            
            self.logger.info(f"Processing PDF: {filename}")
            
            # Check if file already processed (by hash)
            temp_metadata = self.extract_metadata_from_pdf(file_path, filename)
            file_hash = temp_metadata.get("file_hash")
            
            if file_hash != "unknown":
                existing_docs = await self.vector_store.search_by_metadata({"file_hash": file_hash})
                if existing_docs:
                    self.logger.info(f"File {filename} already processed (hash: {file_hash})")
                    return {
                        "status": "already_exists",
                        "filename": filename,
                        "chunks": len(existing_docs),
                        "message": "File already processed"
                    }
            
            # Extract text from PDF
            text = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(), 
                self.extract_text_from_pdf, 
                file_path
            )
            
            if not text.strip():
                raise Exception("No text could be extracted from PDF")
            
            # Extract metadata
            metadata = self.extract_metadata_from_pdf(file_path, filename)
            metadata["text_length"] = len(text)
            metadata["extraction_method"] = "PyPDF2"
            
            # Chunk text
            chunks = self.chunk_text(text, metadata)
            
            if not chunks:
                raise Exception("Failed to create text chunks")
            
            # Add chunks to vector store
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_metadatas = [chunk["metadata"] for chunk in chunks]
            
            success = await self.vector_store.add_documents(
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            if success:
                result = {
                    "status": "success",
                    "filename": filename,
                    "text_length": len(text),
                    "chunks": len(chunks),
                    "pages": metadata.get("page_count", 0),
                    "file_size": metadata.get("file_size", 0),
                    "file_hash": file_hash
                }
                self.logger.info(f"Successfully processed PDF: {filename} ({len(chunks)} chunks)")
                return result
            else:
                raise Exception("Failed to add chunks to vector store")
                
        except Exception as e:
            error_msg = f"PDF processing failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "filename": filename or "unknown",
                "error": str(e),
                "message": error_msg
            }
    
    async def process_multiple_pdfs(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF files concurrently"""
        try:
            # Process files with limited concurrency
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent PDF processing
            
            async def process_with_semaphore(file_path: str):
                async with semaphore:
                    return await self.process_pdf(file_path)
            
            tasks = [process_with_semaphore(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "status": "error",
                        "filename": os.path.basename(file_paths[i]),
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Batch PDF processing failed: {e}")
            return [{"status": "error", "error": str(e)}]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get PDF processing statistics"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                "total_documents": vector_stats.get("total_documents", 0),
                "pdf_documents": len([ft for ft in vector_stats.get("file_types", []) if ft == "pdf"]),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "max_file_size_mb": self.max_file_size // (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get processing stats: {e}")
            return {"error": str(e)}
