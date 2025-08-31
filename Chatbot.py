"""
AI Interviewer - FastAPI Service for Adaptive Technical & Vision-Based Interviews
PROFESSIONAL, CONTEXTUAL, AND NATURAL CONVERSATION FLOW
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from model import RAGModel
from RAG_prompt import RAGPrompts

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    success = initialize_models()
    if not success:
        print("Failed to initialize models")
    yield
    # Shutdown
    pass

app = FastAPI(title="AI Interview Service", description="Adaptive Technical & Vision-Based Interview System", lifespan=lifespan)

# Global variables to store initialized models
rag_model = None
prompts = None
questions = []
document_content = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    question_type: Optional[str] = None
    is_follow_up: bool = False
    session_id: Optional[str] = None
    performance_stats: Optional[dict] = None

def initialize_models():
    """Initialize RAG model and generate questions"""
    global rag_model, prompts, questions, document_content
    
    try:
        # Initialize RAG model
        rag_model = RAGModel()
        
        # Load documents
        if not rag_model.load_documents():
            raise Exception("Failed to load documents. Please check your config.env file.")
        
        # Create embeddings
        if not rag_model.create_embeddings():
            raise Exception("Failed to create embeddings.")
        
        # Get document content
        jd_content, vision_content, resume_content = rag_model.get_document_content()
        if not all([jd_content, vision_content, resume_content]):
            raise Exception("Failed to extract document content.")
        
        document_content = {
            'jd': jd_content,
            'vision': vision_content,
            'resume': resume_content
        }
        
        # Get question weights
        weights = rag_model.get_question_weights()
        
        # Initialize prompts
        prompts = RAGPrompts(rag_model.api_key)
        
        # Generate initial questions
        questions = prompts.generate_weighted_questions(jd_content, vision_content, resume_content, weights)
        
        if not questions:
            questions = [
                "Technical: FEA in Automotive - Walk me through how you would approach a complex finite element analysis for a new vehicle chassis design, including the key challenges and trade-offs you'd consider.",
                "Vision/Mission: Cross-Functional Leadership - Describe a situation where you had to lead a cross-functional team through a challenging project. How did you align everyone's vision and handle conflicting priorities?",
                "Technical: ML Deployment - Explain your approach to deploying a machine learning model in production, including the infrastructure decisions, monitoring strategies, and how you'd handle model drift.",
                "Experience: Problem-Solving - Tell me about the most complex technical problem you've solved. Walk me through your problem-solving process, the alternatives you considered, and the final solution."
            ]
        
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

# Session storage (in production, use proper session management)
sessions = {}
current_question_indices = {}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for AI interviewer"""
    global rag_model, prompts, questions, document_content, sessions, current_question_indices
    
    if not rag_model or not prompts:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    session_id = request.session_id or "default"
    message = request.message.strip()
    
    # Initialize session if not exists
    if session_id not in sessions:
        sessions[session_id] = {
            'total_questions_asked': 0,
            'last_question': None,
            'conversation_history': []
        }
        current_question_indices[session_id] = 0
    
    session = sessions[session_id]
    current_index = current_question_indices[session_id]
    
    # Handle navigation commands
    if message.lower() in ['quit', 'exit', 'stop', 'bye', 'q']:
        return ChatResponse(
            response="Interview ended. Thank you for participating!",
            session_id=session_id
        )
    
    # Handle next question command
    if message.lower() in ['next question', 'next', 'n', 'skip']:
        if current_index < len(questions):
            current_question_indices[session_id] += 1
            current_index = current_question_indices[session_id]
            
            if current_index < len(questions):
                question = questions[current_index]
                session['last_question'] = question
                session['total_questions_asked'] += 1
                
                question_type = "Technical" if "Technical:" in question else "Vision/Mission" if "Vision/Mission:" in question else "Experience"
                
                return ChatResponse(
                    response=f"🔍 QUESTION {session['total_questions_asked']}\n\n{question}",
                    question_type=question_type,
                    session_id=session_id
                )
            else:
                return ChatResponse(
                    response="All prepared questions have been completed. Thank you!",
                    session_id=session_id
                )
        else:
            return ChatResponse(
                response="All questions completed. Thank you!",
                session_id=session_id
            )
    
    # Handle start command or first interaction
    if message.lower() in ['start', 'begin', 'hello', 'hi'] or not session['conversation_history']:
        if current_index < len(questions):
            question = questions[current_index]
            session['last_question'] = question
            session['total_questions_asked'] += 1
            
            question_type = "Technical" if "Technical:" in question else "Vision/Mission" if "Vision/Mission:" in question else "Experience"
            
            return ChatResponse(
                response=f"Welcome to the AI Interview! Let's begin.\n\n🔍 QUESTION {session['total_questions_asked']}\n\n{question}",
                question_type=question_type,
                session_id=session_id
            )
    
    # Handle actual answer - generate follow-up question
    if len(message) > 10 and session['last_question']:
        try:
            follow_up_question = prompts.generate_adaptive_followup(
                session['last_question'], 
                message, 
                document_content['jd'], 
                document_content['vision'], 
                document_content['resume']
            )
            
            session['conversation_history'].append({
                'question': session['last_question'],
                'answer': message,
                'follow_up': follow_up_question
            })
            
            return ChatResponse(
                response=f"Thank you for that detailed response. Here's a follow-up question:\n\n🔍 FOLLOW-UP QUESTION\n\n{follow_up_question}",
                is_follow_up=True,
                session_id=session_id
            )
            
        except Exception:
            # Graceful fallback
            fallback_question = "Can you provide more specific details about the tools, technologies, or approaches you mentioned in your answer?"
            
            return ChatResponse(
                response=f"Thank you for that response. Here's a follow-up question:\n\n🔍 FOLLOW-UP QUESTION\n\n{fallback_question}",
                is_follow_up=True,
                session_id=session_id
            )
    
    # Default response for unclear input
    return ChatResponse(
        response="Please provide a detailed answer, or use 'next question' to move to the next question, or 'start' to begin the interview.",
        session_id=session_id
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)