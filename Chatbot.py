"""
AI Interviewer - Adaptive Technical & Vision-Based Interview System
PROFESSIONAL, CONTEXTUAL, AND NATURAL CONVERSATION FLOW
"""

from model import RAGModel
from RAG_prompt import RAGPrompts
import os
import time

def main():
    """AI Interviewer - Main interview interface with natural conversation flow"""
    start_time = time.time()
    
    print("🎯 AI Interviewer - Adaptive Technical & Vision Interview")
    print("=" * 60)
    
    # Initialize RAG model
    rag_model = RAGModel()
    
    # Load documents
    print("📄 Loading documents...")
    doc_start = time.time()
    if not rag_model.load_documents():
        print("❌ Failed to load documents. Please check your config.env file.")
        return
    doc_time = time.time() - doc_start
    print(f"✅ Documents loaded in {doc_time:.2f}s")
    
    # Create embeddings
    print("🔎 Creating vector database...")
    embedding_start = time.time()
    if not rag_model.create_embeddings():
        print("❌ Failed to create embeddings.")
        return
    embedding_time = time.time() - embedding_start
    print(f"✅ Embeddings created in {embedding_time:.2f}s")
    
    # Get document content
    jd_content, vision_content, resume_content = rag_model.get_document_content()
    if not all([jd_content, vision_content, resume_content]):
        print("❌ Failed to extract document content.")
        return
    
    # Get question weights and technical settings
    weights = rag_model.get_question_weights()
    technical_depth = rag_model.get_technical_depth()
    evaluation_params = rag_model.get_evaluation_parameters()
    
    # Get performance stats
    perf_stats = rag_model.get_performance_stats()
    
    print("✅ AI Interviewer ready!")
    print(f"\n⚡ Performance Stats:")
    print(f"   • Document processing: {perf_stats['processing_time']:.2f}s")
    print(f"   • Chunk size: {perf_stats['chunk_size']} chars")
    print(f"   • Embedding model: {perf_stats['embedding_model']}")
    print(f"   • Technical depth: {perf_stats['technical_depth']}")
    
    print("\n📚 Documents loaded:")
    for i, doc_type in enumerate(rag_model.doc_types):
        print(f"   {i+1}. {doc_type}")
    
    print(f"\n📊 Question Distribution:")
    print(f"   • Technical questions: {weights['JD']}")
    print(f"   • Experience questions: {weights['Resume']}")
    print(f"   • Vision/Mission questions: {weights['Vision_Mission']}")
    
    print(f"\n🔍 Evaluation Parameters:")
    for param in evaluation_params:
        print(f"   • {param.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    print("🎤 Starting AI Interviewer Session...")
    print("=" * 60)
    
    # Initialize prompts
    prompts = RAGPrompts(rag_model.api_key)
    
    # Generate initial questions
    print("\n🤖 AI Interviewer: Generating in-depth, scenario-based questions...")
    question_start = time.time()
    questions = prompts.generate_weighted_questions(jd_content, vision_content, resume_content, weights)
    question_time = time.time() - question_start
    
    if not questions:
        print("❌ Failed to generate questions. Using fallback questions.")
        questions = [
            "Technical: FEA in Automotive - Walk me through how you would approach a complex finite element analysis for a new vehicle chassis design, including the key challenges and trade-offs you'd consider.",
            "Vision/Mission: Cross-Functional Leadership - Describe a situation where you had to lead a cross-functional team through a challenging project. How did you align everyone's vision and handle conflicting priorities?",
            "Technical: ML Deployment - Explain your approach to deploying a machine learning model in production, including the infrastructure decisions, monitoring strategies, and how you'd handle model drift.",
            "Experience: Problem-Solving - Tell me about the most complex technical problem you've solved. Walk me through your problem-solving process, the alternatives you considered, and the final solution."
        ]
    
    print(f"✅ Generated {len(questions)} in-depth questions in {question_time:.2f}s")
    
    # AI Interviewer Session
    current_question_index = 0
    total_questions_asked = 0
    interview_start = time.time()
    
    print("\n" + "=" * 60)
    print("🎤 AI INTERVIEWER SESSION")
    print("=" * 60)
    print("This is an adaptive interview. Questions will flow naturally based on your responses.")
    print("Commands: 'next question', 'stop', 'quit', or 'exit' to end the interview.")
    print("-" * 60)
    
    while True:
        if current_question_index < len(questions):
            current_question = questions[current_question_index]
            total_questions_asked += 1
            
            # Display question with clear heading
            print(f"\n🔍 QUESTION {total_questions_asked}")
            print("=" * 60)
            print(current_question)
            print("=" * 60)
            
            # Get user's answer
            user_answer = input("\n👤 User: ").strip()
            
            # Check for navigation commands first
            if user_answer.lower() in ['quit', 'exit', 'stop', 'bye', 'q']:
                print("\n👋 Interview ended. Thank you!")
                break
            
            # Check for "next question" command
            if user_answer.lower() in ['next question', 'next', 'n', 'skip']:
                print(f"\n🤖 AI Interviewer: Moving to next question...")
                current_question_index += 1
                continue
            
            if not user_answer:
                print("🤖 AI Interviewer: Please provide a detailed answer or use 'next question' to skip.")
                continue
            
            # Only generate follow-up if user provided an actual answer (not a command)
            if len(user_answer) > 10:  # Simple check to ensure it's an actual answer
                print(f"\n🤖 AI Interviewer: Thank you for that detailed response. Let me ask a follow-up question based on what you've shared...")
                
                # Generate contextual follow-up
                follow_up_start = time.time()
                
                try:
                    follow_up_question = prompts.generate_adaptive_followup(
                        current_question, user_answer, jd_content, vision_content, resume_content
                    )
                    follow_up_time = time.time() - follow_up_start
                    
                    print(f"\n🔍 FOLLOW-UP QUESTION (Generated in {follow_up_time:.2f}s)")
                    print("=" * 60)
                    print(follow_up_question)
                    print("=" * 60)
                    
                    # Get follow-up answer
                    follow_up_answer = input("\n👤 User: ").strip()
                    
                    if follow_up_answer.lower() in ['quit', 'exit', 'stop', 'bye', 'q']:
                        print("\n👋 Interview ended. Thank you!")
                        break
                    
                    # Check for "next question" command in follow-up response
                    if follow_up_answer.lower() in ['next question', 'next', 'n', 'skip']:
                        print(f"\n🤖 AI Interviewer: Moving to next question...")
                        current_question_index += 1
                        continue
                    
                    if follow_up_answer:
                        total_questions_asked += 1
                        
                except Exception as e:
                    # Graceful fallback if API fails
                    print(f"\n🔍 FOLLOW-UP QUESTION")
                    print("=" * 60)
                    print("Can you provide more specific details about the tools, technologies, or approaches you mentioned in your answer?")
                    print("=" * 60)
                    
                    # Get follow-up answer
                    follow_up_answer = input("\n👤 User: ").strip()
                    
                    if follow_up_answer.lower() in ['quit', 'exit', 'stop', 'bye', 'q']:
                        print("\n👋 Interview ended. Thank you!")
                        break
                    
                    # Check for "next question" command in follow-up response
                    if follow_up_answer.lower() in ['next question', 'next', 'n', 'skip']:
                        print(f"\n🤖 AI Interviewer: Moving to next question...")
                        current_question_index += 1
                        continue
                    
                    if follow_up_answer:
                        total_questions_asked += 1
            
            # Continue naturally to next question
            print(f"\n🤖 AI Interviewer: Excellent discussion. Let's continue with the next question...")
            current_question_index += 1
            
        else:
            # Generate more questions if needed
            print("\n🤖 AI Interviewer: Generating additional adaptive questions...")
            additional_start = time.time()
            
            try:
                additional_questions = prompts.generate_additional_questions(
                    total_questions_asked, jd_content, vision_content, resume_content
                )
                
                additional_time = time.time() - additional_start
                
                if additional_questions:
                    questions.extend(additional_questions)
                    print(f"✅ Generated {len(additional_questions)} additional questions in {additional_time:.2f}s")
                    continue
                else:
                    print("\n🎉 All questions completed!")
                    break
                    
            except Exception as e:
                # Graceful fallback if API fails
                print("✅ Continuing with existing questions...")
                current_question_index = 0  # Restart with existing questions
                continue
    
    # Interview completed
    total_interview_time = time.time() - interview_start
    total_system_time = time.time() - start_time
    
    print("\n🎯 AI Interviewer Session completed!")
    print("📋 Summary:")
    print(f"   • Total questions asked: {total_questions_asked}")
    print(f"   • Interview time: {total_interview_time:.2f}s")
    print(f"   • Total system time: {total_system_time:.2f}s")
    
    # Performance stats
    prompt_stats = prompts.get_performance_stats()
    print(f"\n⚡ Performance Summary:")
    print(f"   • API calls: {prompt_stats['total_api_calls']}")
    print(f"   • Total API time: {prompt_stats['total_api_time']:.2f}s")
    print(f"   • Average API response: {prompt_stats['avg_api_time']:.2f}s")
    print(f"   • Document processing: {perf_stats['processing_time']:.2f}s")
    print(f"   • Embedding creation: {embedding_time:.2f}s")
    
    print("   • Thank you for participating in this interview!")

if __name__ == "__main__":
    main()
