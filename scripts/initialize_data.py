#!/usr/bin/env python3
"""
Initialize Vector Store with Data Folder Contents

This script processes all PDF files in the data/documents folder
and loads them into the ChromaDB vector store for research.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.vector_store import VectorStoreService
from dotenv import load_dotenv

async def main():
    """Initialize vector store with data folder contents"""
    
    # Load environment variables
    load_dotenv()
    
    print("🔄 Initializing Research Assistant Data Store...")
    print("=" * 50)
    
    # Initialize vector store service
    vector_store = VectorStoreService()
    
    # Check current status
    status = vector_store.get_data_folder_status()
    print(f"📁 Data folder: {status['data_folder']}")
    print(f"📄 PDF files found: {status['pdf_files_found']}")
    print(f"📚 Documents in collection: {status['documents_in_collection']}")
    
    if status['pdf_files_found'] == 0:
        print("\n❌ No PDF files found in data folder!")
        print(f"   Please add PDF files to: {status['data_folder']}")
        return
    
    if status['is_loaded'] and status['documents_in_collection'] > 0:
        print(f"\n📋 Data already loaded ({status['documents_in_collection']} documents)")
        
        response = input("\n🔄 Do you want to reload all data? This will reset the collection. (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("👍 Using existing data.")
            return
        else:
            print("🗑️ Resetting collection...")
            vector_store.reset_collection()
    
    print(f"\n🚀 Processing {status['pdf_files_found']} PDF files...")
    print("-" * 30)
    
    # Initialize from data folder
    result = await vector_store.initialize_from_data_folder()
    
    print("\n" + "=" * 50)
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    
    if result["status"] == "completed":
        print(f"✅ Total files processed: {result['total_files']}")
        print(f"✅ Successful: {result['successful']}")
        print(f"❌ Failed: {result['failed']}")
        
        if result['failed'] > 0:
            print(f"\n❌ Failed files:")
            for res in result['results']:
                if res['status'] == 'error':
                    print(f"   - {res.get('filename', 'unknown')}: {res.get('error', 'unknown error')}")
        
        # Final status
        final_status = vector_store.get_data_folder_status()
        print(f"\n📚 Total documents in collection: {final_status['documents_in_collection']}")
        print("\n🎉 Data initialization completed!")
        print("🚀 You can now start the research assistant.")
        
    else:
        print(f"❌ Initialization failed: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
