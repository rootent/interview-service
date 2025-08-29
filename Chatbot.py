"""
Interview Chatbot - Main interface for the AI-powered interview system
OPTIMIZED FOR SPEED
"""

from model import RAGModel
from RAG_prompt import RAGPrompts
import os
import time

def main():
    """Main interview interface - OPTIMIZED FOR SPEED"""
    start_time = time.time()
    
    print("🎯 Interview RAG Chatbot - AI-Powered Interview System")
    print("⚡ OPTIMIZED FOR SPEED")
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
    
    # Get question weights
    weights = rag_model.get_question_weights()
    
    # Get performance stats
    perf_stats = rag_model.get_performance_stats()
    
    print("✅ Interview system ready!")
    print(f"\n⚡ Performance Stats:")
    print(f"   • Document processing: {perf_stats['processing_time']:.2f}s")
    print(f"   • Chunk size: {perf_stats['chunk_size']} chars")
    print(f"   • Content length: {perf_stats['max_content_length']} chars")
    print(f"   • Embedding model: {perf_stats['embedding_model']}")
    
    print("\n📚 Documents loaded:")
    for i, doc_type in enumerate(rag_model.doc_types):
        print(f"   {i+1}. {doc_type}")
    
    print(f"\n📊 Question Distribution:")
    print(f"   • JD-focused questions: {weights['JD']}")
    print(f"   • Resume/Experience questions: {weights['Resume']}")
    print(f"   • Vision/Mission questions: {weights['Vision_Mission']}")
    
    print("\n" + "=" * 60)
    print("🎤 Starting AI-Powered Interview...")
    print("=" * 60)
    
    # Initialize prompts
    prompts = RAGPrompts(rag_model.api_key)
    
    # Generate weighted questions
    print("\n🤖 AI Interviewer: Generating weighted interview questions...")
    question_start = time.time()
    questions = prompts.generate_weighted_questions(jd_content, vision_content, resume_content, weights)
    question_time = time.time() - question_start
    
    if not questions:
        print("❌ Failed to generate questions. Using fallback questions.")
        questions = [
            "Q1: Tell me about your relevant experience for this role.",
            "Q2: How do you approach problem-solving in your work?",
            "Q3: What interests you about our company's vision?",
            "Q4: Describe a challenging project you've worked on.",
            "Q5: How do you stay updated with industry trends?"
        ]
    
    print(f"✅ Generated {len(questions)} weighted interview questions in {question_time:.2f}s")
    
    # Interview loop - OPTIMIZED
    current_question_index = 0
    total_follow_ups = 0
    interview_start = time.time()
    
    while current_question_index < len(questions):
        current_question = questions[current_question_index]
        
        # Ask the main question
        print(f"\n🤖 AI Interviewer: {current_question}")
        
        # Get candidate's answer
        candidate_answer = input("\n👤 Candidate: ").strip()
        
        if candidate_answer.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\n👋 Interview ended. Thank you!")
            break
        
        if not candidate_answer:
            print("🤖 AI Interviewer: Please provide an answer to continue.")
            continue
        
        # Evaluate the answer - OPTIMIZED
        eval_start = time.time()
        evaluation = prompts.evaluate_answer_quality(current_question, candidate_answer, jd_content, vision_content, resume_content)
        eval_time = time.time() - eval_start
        print(f"\n📊 Answer Evaluation ({eval_time:.2f}s): {evaluation}")
        
        # Follow-up questions loop - OPTIMIZED (reduced follow-ups)
        follow_up_count = 0
        max_follow_ups = rag_model.max_follow_ups  # Reduced from 3 to 2
        
        while follow_up_count < max_follow_ups:
            follow_up_count += 1
            
            # Generate follow-up question - OPTIMIZED
            follow_up_start = time.time()
            follow_up = prompts.generate_follow_up_question(
                current_question, candidate_answer, jd_content, vision_content, resume_content
            )
            follow_up_time = time.time() - follow_up_start
            
            print(f"\n🤖 AI Interviewer (Follow-up {follow_up_count}, {follow_up_time:.2f}s): {follow_up}")
            
            # Get follow-up answer
            follow_up_answer = input("\n👤 Candidate: ").strip()
            
            if follow_up_answer.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Interview ended. Thank you!")
                return
            
            if follow_up_answer.lower() in ['no idea', 'i don\'t know', 'not sure', 'unsure', 'no clue', 'skip']:
                print(f"\n🤖 AI Interviewer: Thank you for your honesty. Let's move to the next question.")
                break
            
            if not follow_up_answer:
                print("🤖 AI Interviewer: Please provide an answer to continue.")
                continue
            
            # Evaluate follow-up answer - OPTIMIZED
            follow_up_eval_start = time.time()
            follow_up_evaluation = prompts.evaluate_answer_quality(
                follow_up, follow_up_answer, jd_content, vision_content, resume_content
            )
            follow_up_eval_time = time.time() - follow_up_eval_start
            print(f"\n📊 Follow-up Evaluation ({follow_up_eval_time:.2f}s): {follow_up_evaluation}")
            
            # Update for next iteration
            current_question = follow_up
            candidate_answer = follow_up_answer
        
        total_follow_ups += follow_up_count
        
        # Move to next main question
        current_question_index += 1
        
        if current_question_index < len(questions):
            print(f"\n🔄 Moving to question {current_question_index + 1} of {len(questions)}...")
            print("-" * 40)
    
    # Interview completed
    total_interview_time = time.time() - interview_start
    total_system_time = time.time() - start_time
    
    print("\n🎉 Interview completed!")
    print("📋 Summary:")
    print(f"   • Total questions asked: {len(questions)}")
    print(f"   • Total follow-up questions: {total_follow_ups}")
    print(f"   • Question distribution: JD ({weights['JD']}), Resume ({weights['Resume']}), Vision ({weights['Vision_Mission']})")
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
    
    print("   • Thank you for participating in this AI-powered interview!")

if __name__ == "__main__":
    main()
