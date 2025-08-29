"""
RAG Prompts - Handles prompt generation for interview questions and follow-ups
OPTIMIZED FOR SPEED
"""

import google.generativeai as genai
import time

class RAGPrompts:
    def __init__(self, api_key):
        """Initialize with Gemini API key - OPTIMIZED FOR SPEED"""
        genai.configure(api_key=api_key)
        # Use the fastest Gemini model available
        self.model = genai.GenerativeModel("gemini-1.5-flash")  # Faster than 2.5-flash
        self._api_calls = 0
        self._total_time = 0
    
    def _timed_api_call(self, prompt, max_tokens=150):
        """Make a timed API call with token limits for speed"""
        start_time = time.time()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,  # Limit output for speed
                    temperature=0.7,  # Slightly higher for variety
                    top_p=0.9
                )
            )
            self._api_calls += 1
            self._total_time += time.time() - start_time
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ API call failed: {e}")
            return None
    
    def generate_weighted_questions(self, jd_content, vision_content, resume_content, weights):
        """Generate questions with weighting - OPTIMIZED FOR SPEED"""
        start_time = time.time()
        
        all_questions = []
        
        # Generate JD-focused questions - OPTIMIZED
        jd_prompt = f"""Generate {weights['JD']} practical interview questions based on this JD:

JD: {jd_content[:800]}

Create {weights['JD']} questions that:
1. Are conversational and natural
2. Can be answered in 2-3 sentences
3. Test practical knowledge
4. Ask for specific examples

Format: Q1: [question] Q2: [question] etc.
"""
        
        jd_response = self._timed_api_call(jd_prompt, max_tokens=100)
        if jd_response:
            jd_questions_list = [q.strip() for q in jd_response.split('Q') if q.strip() and ':' in q]
            all_questions.extend([f"Q{q}" for q in jd_questions_list[:weights['JD']]])
        
        # Generate Resume/Experience-focused questions - OPTIMIZED
        resume_prompt = f"""Generate {weights['Resume']} experience questions based on this resume:

Resume: {resume_content[:800]}

Create {weights['Resume']} questions that:
1. Ask for concrete examples
2. Test practical experience
3. Feel natural in interviews
4. Can be answered briefly

Format: Q1: [question] Q2: [question] etc.
"""
        
        resume_response = self._timed_api_call(resume_prompt, max_tokens=100)
        if resume_response:
            resume_questions_list = [q.strip() for q in resume_response.split('Q') if q.strip() and ':' in q]
            all_questions.extend([f"Q{q}" for q in resume_questions_list[:weights['Resume']]])
        
        # Generate Vision/Mission-focused questions - OPTIMIZED
        vision_prompt = f"""Generate {weights['Vision_Mission']} cultural questions based on this vision:

Vision: {vision_content[:500]}

Create {weights['Vision_Mission']} questions that:
1. Test cultural fit
2. Can be answered briefly
3. Feel natural
4. Assess values alignment

Format: Q1: [question] Q2: [question] etc.
"""
        
        vision_response = self._timed_api_call(vision_prompt, max_tokens=100)
        if vision_response:
            vision_questions_list = [q.strip() for q in vision_response.split('Q') if q.strip() and ':' in q]
            all_questions.extend([f"Q{q}" for q in vision_questions_list[:weights['Vision_Mission']]])
        
        # Fallback questions if API fails
        if not all_questions:
            all_questions = [
                "Q1: Tell me about your relevant experience for this role.",
                "Q2: How do you approach problem-solving?",
                "Q3: What interests you about our company?",
                "Q4: Describe a challenging project you've worked on.",
                "Q5: How do you stay updated with industry trends?"
            ]
        
        total_time = time.time() - start_time
        print(f"⚡ Generated {len(all_questions)} questions in {total_time:.2f}s ({self._api_calls} API calls)")
        
        return all_questions
    
    def generate_follow_up_question(self, original_question, candidate_answer, jd_content, vision_content, resume_content):
        """Generate a follow-up question - OPTIMIZED FOR SPEED"""
        
        # Determine the focus area of the original question
        focus_area = "JD"  # Default to JD focus
        
        if any(word in original_question.lower() for word in ['vision', 'mission', 'culture', 'values', 'company']):
            focus_area = "Vision_Mission"
        elif any(word in original_question.lower() for word in ['experience', 'project', 'work', 'background', 'resume']):
            focus_area = "Resume"
        
        # Generate appropriate follow-up based on focus area - OPTIMIZED
        if focus_area == "JD":
            prompt = f"""Generate a simple follow-up question:

Question: {original_question}
Answer: {candidate_answer}
Context: {jd_content[:600]}

Create a follow-up that:
1. Is simple and conversational
2. Asks for a specific example
3. Can be answered briefly

Return only the question.
"""
        elif focus_area == "Resume":
            prompt = f"""Generate a focused follow-up question:

Question: {original_question}
Answer: {candidate_answer}
Context: {resume_content[:600]}

Create a follow-up that:
1. Is specific and focused
2. Asks for concrete details
3. Can be answered briefly

Return only the question.
"""
        else:
            prompt = f"""Generate a simple cultural follow-up:

Question: {original_question}
Answer: {candidate_answer}
Context: {vision_content[:400]}

Create a follow-up that:
1. Tests cultural understanding
2. Can be answered briefly
3. Feels natural

Return only the question.
"""
        
        response = self._timed_api_call(prompt, max_tokens=80)
        return response if response else "Can you give me a specific example of that?"
    
    def evaluate_answer_quality(self, question, answer, jd_content, vision_content, resume_content):
        """Evaluate the quality of the candidate's answer - OPTIMIZED FOR SPEED"""
        
        # Determine question type for appropriate evaluation criteria
        if any(word in question.lower() for word in ['vision', 'mission', 'culture', 'values', 'company']):
            evaluation_focus = "Vision/Mission alignment"
        elif any(word in question.lower() for word in ['experience', 'project', 'work', 'background']):
            evaluation_focus = "Experience and practical knowledge"
        else:
            evaluation_focus = "Technical skills and role requirements"
        
        # Much shorter prompt for faster processing
        prompt = f"""Rate this interview answer (1-10) with brief feedback:

Q: {question}
A: {answer}
Focus: {evaluation_focus}

Context: {jd_content[:400]} | {vision_content[:300]} | {resume_content[:400]}

Format: Score: X/10 - Brief feedback
"""
        
        response = self._timed_api_call(prompt, max_tokens=60)
        return response if response else "Score: 7/10 - Good answer, could use more specific examples"
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'total_api_calls': self._api_calls,
            'total_api_time': self._total_time,
            'avg_api_time': self._total_time / max(self._api_calls, 1)
        }
