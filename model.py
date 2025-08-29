"""
RAG Model - Core functionality for document processing and embeddings
OPTIMIZED FOR SPEED
"""

import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pickle
import hashlib
from pathlib import Path
import time

# Load environment variables
load_dotenv('config.env')

class RAGModel:
    def __init__(self):
        """Initialize RAG model with configuration - OPTIMIZED FOR SPEED"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.jd_path = os.getenv('JD_PATH')
        self.vision_path = os.getenv('VISION_MISSION_PATH')
        self.resume_path = os.getenv('RESUME_PATH')
        
        # Question weights
        self.jd_weight = float(os.getenv('JD_WEIGHT', 0.5))
        self.resume_weight = float(os.getenv('RESUME_WEIGHT', 0.35))
        self.vision_weight = float(os.getenv('VISION_WEIGHT', 0.15))
        
        # Model configuration - OPTIMIZED FOR SPEED
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L3-v2')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 300))  # Much smaller for speed
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))  # Minimal overlap
        self.max_follow_ups = int(os.getenv('MAX_FOLLOW_UPS', 2))  # Reduced follow-ups
        
        # Performance settings
        self.use_cache = os.getenv('USE_CACHE', 'true').lower() == 'true'
        self.max_content_length = int(os.getenv('MAX_CONTENT_LENGTH', 1000))
        self.batch_size = int(os.getenv('BATCH_SIZE', 3))
        
        # Cache configuration
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize Gemini with faster model
        genai.configure(api_key=self.api_key)
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.docs = None
        self.doc_types = None
        self._document_hashes = {}
        self._processing_time = 0
        
    def _get_file_hash(self, file_path):
        """Get file hash for caching"""
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        return f"{file_path}_{stat.st_mtime}_{stat.st_size}"
    
    def _load_from_cache(self):
        """Load processed documents from cache if available"""
        if not self.use_cache:
            return False
            
        cache_file = self.cache_dir / "processed_docs.pkl"
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if all files are still the same
            current_hashes = {
                "jd": self._get_file_hash(self.jd_path),
                "vision": self._get_file_hash(self.vision_path),
                "resume": self._get_file_hash(self.resume_path)
            }
            
            if cached_data.get('hashes') == current_hashes:
                self.docs = cached_data['docs']
                self.doc_types = cached_data['doc_types']
                self.embeddings = cached_data['embeddings']
                self.vectorstore = cached_data['vectorstore']
                print("✅ Loaded from cache - documents already processed")
                return True
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
        
        return False
    
    def _save_to_cache(self):
        """Save processed documents to cache"""
        if not self.use_cache:
            return
            
        try:
            cache_file = self.cache_dir / "processed_docs.pkl"
            cache_data = {
                'docs': self.docs,
                'doc_types': self.doc_types,
                'embeddings': self.embeddings,
                'vectorstore': self.vectorstore,
                'hashes': {
                    "jd": self._get_file_hash(self.jd_path),
                    "vision": self._get_file_hash(self.vision_path),
                    "resume": self._get_file_hash(self.resume_path)
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("💾 Saved to cache for future use")
        except Exception as e:
            print(f"⚠️ Cache saving failed: {e}")
    
    def load_documents(self):
        """Load documents from specified paths with caching - OPTIMIZED"""
        start_time = time.time()
        
        # Try to load from cache first
        if self._load_from_cache():
            self._processing_time = time.time() - start_time
            return True
        
        all_docs = []
        doc_types = []
        
        document_paths = {
            "JD": self.jd_path,
            "Vision_Mission": self.vision_path,
            "Resume": self.resume_path
        }
        
        for doc_type, file_path in document_paths.items():
            if not os.path.exists(file_path):
                print(f"❌ {doc_type} file not found: {file_path}")
                return False
            
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
                doc_types.append(doc_type)
                print(f"✅ Loaded {doc_type}: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {doc_type} ({file_path}): {e}")
                return False
        
        self.docs = all_docs
        self.doc_types = doc_types
        
        self._processing_time = time.time() - start_time
        return True
    
    def create_embeddings(self):
        """Create embeddings and vector database with caching - OPTIMIZED FOR SPEED"""
        if not self.docs:
            print("❌ No documents loaded")
            return False
        
        # Check if we already have embeddings from cache
        if self.embeddings and self.vectorstore:
            print("✅ Using cached embeddings")
            return True
        
        start_time = time.time()
        
        # Split documents into chunks - OPTIMIZED for speed
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(self.docs)
        print(f"📝 Created {len(chunks)} chunks (optimized: {self.chunk_size} chars)")
        
        # Use the fastest embedding model available
        try:
            # Use the lightest model for maximum speed
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Fastest available
                model_kwargs={'device': 'cpu'},  # Force CPU for compatibility
                encode_kwargs={'batch_size': self.batch_size}  # Batch processing
            )
            print("🚀 Using ultra-fast embedding model: paraphrase-MiniLM-L3-v2")
        except Exception as e:
            print(f"⚠️ Fast model failed, using fallback: {e}")
            # Fallback to original model
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        # Create vector database with optimized settings
        self.vectorstore = Chroma.from_documents(
            chunks, 
            embedding=self.embeddings, 
            persist_directory="chroma_db"
        )
        
        # Save to cache
        self._save_to_cache()
        
        embedding_time = time.time() - start_time
        print(f"🔎 Vector database created successfully in {embedding_time:.2f}s")
        return True
    
    def get_document_content(self):
        """Get content from loaded documents - OPTIMIZED for API calls"""
        if not self.docs or len(self.docs) < 3:
            return None, None, None
        
        # Extract content and truncate for faster API calls - MUCH SHORTER
        jd_content = self.docs[0].page_content[:self.max_content_length] if len(self.docs) > 0 else ""
        vision_content = self.docs[1].page_content[:self.max_content_length//2] if len(self.docs) > 1 else ""  # Even shorter
        resume_content = self.docs[2].page_content[:self.max_content_length] if len(self.docs) > 2 else ""
        
        return jd_content, vision_content, resume_content
    
    def get_document_summaries(self):
        """Get brief summaries for faster processing - OPTIMIZED"""
        if not self.docs or len(self.docs) < 3:
            return None, None, None
        
        # Create very brief summaries for faster API calls
        jd_summary = self.docs[0].page_content[:500] + "..." if len(self.docs) > 0 else ""
        vision_summary = self.docs[1].page_content[:300] + "..." if len(self.docs) > 1 else ""
        resume_summary = self.docs[2].page_content[:500] + "..." if len(self.docs) > 2 else ""
        
        return jd_summary, vision_summary, resume_summary
    
    def calculate_relevance_score(self, query_embedding, doc_embeddings):
        """Calculate relevance score between query and documents - OPTIMIZED"""
        if not doc_embeddings:
            return 0.0
        
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = cosine_similarity([query_embedding], [doc_emb])[0][0]
            similarities.append(similarity)
        
        return max(similarities) if similarities else 0.0
    
    def get_question_weights(self):
        """Get question distribution based on weights - OPTIMIZED"""
        total_questions = 5  # Reduced from 7 for faster processing
        jd_questions = max(1, int(total_questions * self.jd_weight))
        resume_questions = max(1, int(total_questions * self.resume_weight))
        vision_questions = max(1, int(total_questions * self.vision_weight))
        
        return {
            "JD": jd_questions,
            "Resume": resume_questions,
            "Vision_Mission": vision_questions
        }
    
    def is_initialized(self):
        """Check if model is properly initialized"""
        return (
            self.embeddings is not None and 
            self.vectorstore is not None and 
            self.docs is not None
        )
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'processing_time': self._processing_time,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_content_length': self.max_content_length,
            'embedding_model': self.embedding_model
        }
