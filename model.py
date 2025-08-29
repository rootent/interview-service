"""
RAG Model - Core functionality for document processing and embeddings
OPTIMIZED FOR SPEED & TECHNICAL DEPTH
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
import glob

# Load environment variables
load_dotenv('config.env')

class RAGModel:
    def __init__(self):
        """Initialize RAG model with configuration - OPTIMIZED FOR SPEED & TECHNICAL DEPTH"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.jd_path = os.getenv('JD_PATH')
        self.vision_path = os.getenv('VISION_MISSION_PATH')
        self.resume_path = os.getenv('RESUME_PATH')
        
        # Question weights
        self.jd_weight = float(os.getenv('JD_WEIGHT', 0.5))
        self.resume_weight = float(os.getenv('RESUME_WEIGHT', 0.35))
        self.vision_weight = float(os.getenv('VISION_WEIGHT', 0.15))
        
        # Model configuration - OPTIMIZED FOR SPEED & QUALITY
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L3-v2')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 300))  # Optimized for speed
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))  # Minimal overlap
        self.max_follow_ups = int(os.getenv('MAX_FOLLOW_UPS', 3))  # Increased for technical depth
        
        # Performance settings
        self.use_cache = os.getenv('USE_CACHE', 'true').lower() == 'true'
        self.batch_size = int(os.getenv('BATCH_SIZE', 3))
        
        # Technical interview settings
        self.technical_depth = os.getenv('TECHNICAL_DEPTH', 'high')
        self.evaluation_parameters = os.getenv('EVALUATION_PARAMETERS', 'language,tech_skills,content_depth,problem_solving,communication').split(',')
        
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
    
    def _get_folder_hash(self, folder_path):
        """Get folder hash for caching - handles multiple PDFs"""
        if not os.path.exists(folder_path):
            return None
        
        if os.path.isfile(folder_path):
            # If it's a file, use regular file hash
            return self._get_file_hash(folder_path)
        
        # If it's a folder, hash all PDF files in it
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            return None
        
        # Sort files for consistent hashing
        pdf_files.sort()
        folder_hash = f"folder_{folder_path}"
        
        for pdf_file in pdf_files:
            file_hash = self._get_file_hash(pdf_file)
            if file_hash:
                folder_hash += f"_{file_hash}"
        
        return folder_hash
    
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
                "vision": self._get_folder_hash(self.vision_path),
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
                    "vision": self._get_folder_hash(self.vision_path),
                    "resume": self._get_file_hash(self.resume_path)
                }
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("💾 Saved to cache for future use")
        except Exception as e:
            print(f"⚠️ Cache saving failed: {e}")
    
    def _load_pdf_from_path(self, path, doc_type):
        """Load PDF(s) from path - handles both single files and folders"""
        docs = []
        
        if os.path.isfile(path):
            # Single PDF file
            try:
                loader = PyPDFLoader(path)
                file_docs = loader.load()
                docs.extend(file_docs)
                print(f"✅ Loaded {doc_type}: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Error loading {doc_type} ({path}): {e}")
                return None
        elif os.path.isdir(path):
            # Folder - load all PDFs
            pdf_files = glob.glob(os.path.join(path, "*.pdf"))
            if not pdf_files:
                print(f"❌ No PDF files found in {doc_type} folder: {path}")
                return None
            
            print(f"📁 Found {len(pdf_files)} PDF files in {doc_type} folder")
            for pdf_file in sorted(pdf_files):
                try:
                    loader = PyPDFLoader(pdf_file)
                    file_docs = loader.load()
                    docs.extend(file_docs)
                    print(f"  ✅ Loaded: {os.path.basename(pdf_file)}")
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not load {os.path.basename(pdf_file)}: {e}")
                    continue
            
            if not docs:
                print(f"❌ No PDFs could be loaded from {doc_type} folder")
                return None
                
            print(f"✅ Successfully loaded {len(docs)} document chunks from {doc_type} folder")
        else:
            print(f"❌ {doc_type} path not found: {path}")
            return None
        
        return docs
    
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
        
        for doc_type, path in document_paths.items():
            docs = self._load_pdf_from_path(path, doc_type)
            if docs is None:
                return False
            
            all_docs.extend(docs)
            doc_types.append(doc_type)
        
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
        """Get content from loaded documents - NO LENGTH RESTRICTIONS for quality"""
        if not self.docs or len(self.docs) < 3:
            return None, None, None
        
        # Extract full content without truncation for comprehensive AI responses
        jd_content = self.docs[0].page_content if len(self.docs) > 0 else ""
        
        # Handle vision content - combine all vision documents
        vision_content = ""
        vision_start_index = 1  # Vision documents start after JD
        
        # Find where vision documents end (before resume documents)
        vision_end_index = len(self.docs) - 1  # Default to last document
        
        # Look for resume documents (they come after vision)
        for i, doc in enumerate(self.docs):
            if hasattr(doc, 'metadata') and doc.metadata.get('source', '').endswith('.pdf'):
                # This is a heuristic - resume is typically the last document type
                if i > vision_start_index:
                    vision_end_index = i - 1
                    break
        
        # Combine all vision document content
        for i in range(vision_start_index, vision_end_index + 1):
            if i < len(self.docs):
                vision_content += self.docs[i].page_content + "\n\n"
        
        # Resume content is the last document
        resume_content = self.docs[-1].page_content if len(self.docs) > 0 else ""
        
        return jd_content, vision_content, resume_content
    
    def get_document_summaries(self):
        """Get comprehensive summaries for technical depth - NO LENGTH RESTRICTIONS"""
        if not self.docs or len(self.docs) < 3:
            return None, None, None
        
        # Create comprehensive summaries for technical depth
        jd_summary = self.docs[0].page_content if len(self.docs) > 0 else ""
        
        # Handle vision content - combine all vision documents
        vision_summary = ""
        vision_start_index = 1  # Vision documents start after JD
        
        # Find where vision documents end (before resume documents)
        vision_end_index = len(self.docs) - 1  # Default to last document
        
        # Look for resume documents (they come after vision)
        for i, doc in enumerate(self.docs):
            if hasattr(doc, 'metadata') and doc.metadata.get('source', '').endswith('.pdf'):
                # This is a heuristic - resume is typically the last document type
                if i > vision_start_index:
                    vision_end_index = i - 1
                    break
        
        # Combine all vision document content
        for i in range(vision_start_index, vision_end_index + 1):
            if i < len(self.docs):
                vision_summary += self.docs[i].page_content + "\n\n"
        
        # Resume summary is the last document
        resume_summary = self.docs[-1].page_content if len(self.docs) > 0 else ""
        
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
        """Get question distribution based on weights - OPTIMIZED for technical depth"""
        total_questions = 6  # Increased for technical depth
        jd_questions = max(2, int(total_questions * self.jd_weight))  # Minimum 2 technical questions
        resume_questions = max(2, int(total_questions * self.resume_weight))  # Minimum 2 experience questions
        vision_questions = max(1, int(total_questions * self.vision_weight))  # Minimum 1 cultural question
        
        return {
            "JD": jd_questions,
            "Resume": resume_questions,
            "Vision_Mission": vision_questions
        }
    
    def get_evaluation_parameters(self):
        """Get technical evaluation parameters"""
        return self.evaluation_parameters
    
    def get_technical_depth(self):
        """Get technical depth setting"""
        return self.technical_depth
    
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
            'embedding_model': self.embedding_model,
            'technical_depth': self.technical_depth,
            'evaluation_parameters': self.evaluation_parameters
        }
