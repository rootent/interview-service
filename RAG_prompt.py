"""
RAG Prompts - AI Interviewer Prompt Generation
CONTEXTUAL, ENTITY-ANCHORED FOLLOW-UPS WITH NO GENERIC FILLER
"""

import google.generativeai as genai
import time

class RAGPrompts:
    def __init__(self, api_key):
        """Initialize with Gemini API key - OPTIMIZED FOR CONTEXTUAL FOLLOW-UPS"""
        genai.configure(api_key=api_key)
        # Use the fastest Gemini model available
        self.model = genai.GenerativeModel("gemini-1.5-flash")  # Faster than 2.5-flash
        self._api_calls = 0
        self._total_time = 0
    
    def _timed_api_call(self, prompt, max_tokens=300):
        """Make a timed API call with increased tokens for in-depth questions"""
        start_time = time.time()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,  # Increased for in-depth questions
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
        """Generate in-depth, scenario-based questions with clear labeling"""
        start_time = time.time()
        
        all_questions = []
        
        # Generate Technical questions with clear labeling
        jd_prompt = f"""Generate {weights['JD']} in-depth, scenario-based technical questions based on this Job Description:

JOB DESCRIPTION:
{jd_content}

Create {weights['JD']} questions that:
1. Are labeled clearly (e.g., "Technical: FEA in Automotive", "Technical: ML Deployment")
2. Frame questions around real-world challenges and scenarios
3. Require problem-solving steps, trade-offs, and examples
4. Go beyond theory to practical application
5. Test deep technical knowledge and decision-making
6. Are conversational and professional like a real interview

Examples of good technical questions:
- "Technical: FEA in Automotive - Walk me through how you would approach a complex finite element analysis for a new vehicle chassis design, including the key challenges and trade-offs you'd consider."
- "Technical: ML Deployment - Explain your approach to deploying a machine learning model in production, including the infrastructure decisions, monitoring strategies, and how you'd handle model drift."

Format: [Category]: [Specific Topic] - [Detailed scenario-based question]
"""
        
        jd_response = self._timed_api_call(jd_prompt, max_tokens=250)
        if jd_response:
            jd_questions_list = [q.strip() for q in jd_response.split('\n') if q.strip() and ':' in q]
        all_questions.extend(jd_questions_list[:weights['JD']])
        
        # Generate Experience questions with clear labeling
        resume_prompt = f"""Generate {weights['Resume']} in-depth, scenario-based experience questions based on this resume:

RESUME:
{resume_content}

Create {weights['Resume']} questions that:
1. Are labeled clearly (e.g., "Experience: Problem-Solving", "Experience: Project Management")
2. Ask for specific real-world examples and scenarios
3. Require detailed explanations of challenges and solutions
4. Test practical experience and decision-making
5. Are conversational and professional like a real interview
6. Focus on concrete achievements and learnings

Examples of good experience questions:
- "Experience: Problem-Solving - Tell me about the most complex technical problem you've solved. Walk me through your problem-solving process, the alternatives you considered, and the final solution."
- "Experience: Cross-Functional Collaboration - Describe a situation where you had to work with multiple teams to deliver a project. What challenges did you face and how did you overcome them?"

Format: [Category]: [Specific Topic] - [Detailed scenario-based question]
"""
        
        resume_response = self._timed_api_call(resume_prompt, max_tokens=250)
        if resume_response:
            resume_questions_list = [q.strip() for q in resume_response.split('\n') if q.strip() and ':' in q]
        all_questions.extend(resume_questions_list[:weights['Resume']])
        
        # Generate Vision/Mission questions with clear labeling - CORRECTLY SOURCED FROM VISION DOCUMENT
        vision_prompt = f"""Generate {weights['Vision_Mission']} in-depth, scenario-based vision/mission questions based on this VISION & MISSION document:

VISION & MISSION DOCUMENT:
{vision_content}

Create {weights['Vision_Mission']} questions that:
1. Are labeled clearly (e.g., "Vision/Mission: Values Alignment", "Vision/Mission: Strategic Leadership")
2. Test understanding of the company's ACTUAL vision, mission, and values as stated in the document
3. Present situational challenges requiring vision-driven responses
4. Are conversational and professional like a real interview
5. Focus on leadership, culture, strategic thinking, and values alignment
6. Reference specific elements from the vision document (NOT from resume)
7. Ask for examples of how to apply the company's stated vision/mission in practice

IMPORTANT: Base questions ONLY on what's actually stated in the vision document. Do NOT reference resume content or make assumptions about company values not explicitly mentioned.

Examples of good vision/mission questions:
- "Vision/Mission: Values Alignment - Based on the company's stated vision of [specific vision from document], describe a situation where you had to make a decision that tested your alignment with these values. How did you ensure your choice reflected the company's mission?"
- "Vision/Mission: Strategic Leadership - Given the company's mission to [specific mission from document], tell me about a time when you had to lead a team through a challenging project while maintaining alignment with this mission."

Format: [Category]: [Specific Topic] - [Detailed scenario-based question]
"""
        
        vision_response = self._timed_api_call(vision_prompt, max_tokens=250)
        if vision_response:
            vision_questions_list = [q.strip() for q in vision_response.split('\n') if q.strip() and ':' in q]
        all_questions.extend(vision_questions_list[:weights['Vision_Mission']])
        
        # Fallback questions if API fails
        if not all_questions:
            all_questions = [
                "Technical: FEA in Automotive - Walk me through how you would approach a complex finite element analysis for a new vehicle chassis design, including the key challenges and trade-offs you'd consider.",
                "Vision/Mission: Values Alignment - Based on the company's stated vision and mission, describe a situation where you had to balance project demands with company values. What was the challenge and how did you navigate it?",
                "Technical: ML Deployment - Explain your approach to deploying a machine learning model in production, including the infrastructure decisions, monitoring strategies, and how you'd handle model drift.",
                "Experience: Problem-Solving - Tell me about the most complex technical problem you've solved. Walk me through your problem-solving process, the alternatives you considered, and the final solution."
            ]
        
        total_time = time.time() - start_time
        print(f"⚡ Generated {len(all_questions)} in-depth, scenario-based questions in {total_time:.2f}s ({self._api_calls} API calls)")
        
        return all_questions
    
    def generate_adaptive_followup(self, original_question, user_answer, jd_content, vision_content, resume_content):
        """Generate a specific, contextual follow-up question anchored to entities/tools mentioned by the user"""
        
        prompt = f"""You are an AI Interviewer. Generate a follow-up question that is ALWAYS contextual and anchored to specific entities, tools, or materials mentioned by the user.

ORIGINAL QUESTION: {original_question}
USER'S ANSWER: {user_answer}

JOB & VISION CONTEXT:
JD: {jd_content[:1500]}
Vision: {vision_content[:1000]}
Resume: {resume_content[:1000]}

CRITICAL REQUIREMENTS:
1. NEVER use generic phrases like "Can you be more specific?" or "Can you provide more details?"
2. ALWAYS anchor your follow-up to something concrete the user mentioned
3. Extract specific entities: software (Siemens NX, MATLAB), materials (Mg alloy, carbon fiber), tools (CAD, FEA), techniques, regulations, metrics, etc.
4. Make the follow-up feel like a natural continuation of their answer
5. Probe deeper into the specific technical or experiential details they shared

EXAMPLES OF GOOD CONTEXTUAL FOLLOW-UPS:
- User mentions "Siemens NX" → "Walk me through the specific workflow you use in Siemens NX for this type of analysis, particularly the boundary condition setup you described."
- User mentions "Mg alloy" → "What specific properties of the Mg alloy influenced your design decisions, and how did you validate those material characteristics?"
- User mentions "30% improvement" → "How did you measure and validate that 30% performance improvement? What metrics and testing protocols did you use?"
- User mentions "Agile methodology" → "In that Agile environment, how did you handle the sprint planning and stakeholder communication challenges you mentioned?"

EXAMPLES OF BAD GENERIC FOLLOW-UPS (NEVER USE):
- "Can you provide more details?"
- "Can you be more specific?"
- "Tell me more about that"
- "What else can you share?"

Return only the contextual follow-up question, no additional text.
"""
        
        response = self._timed_api_call(prompt, max_tokens=250)
        return response if response else "What specific technical details or implementation steps did you use for the approach you just described?"
    
    def generate_deep_technical_followup(self, original_question, user_answer, jd_content, vision_content, resume_content):
        """Generate a deep technical follow-up question based on user's answer"""
        
        prompt = f"""Generate a deep technical follow-up question based on this main question and user's answer:

MAIN QUESTION: {original_question}
USER'S ANSWER: {user_answer}

JOB CONTEXT:
JD: {jd_content[:1500]}

Create a follow-up that:
1. Probes deeper into the technical aspects mentioned in their answer
2. Asks for specific technical details, implementation steps, or code examples
3. Tests understanding of underlying technical concepts and trade-offs
4. Requires demonstration of practical technical expertise
5. Is conversational and professional like a real interview
6. Can be answered with technical depth and examples

Examples of good technical follow-ups:
- "Can you walk me through the specific implementation details of that solution?"
- "What were the performance implications of that approach? How did you measure and optimize it?"
- "How would you handle edge cases or error scenarios in that implementation?"

Return only the follow-up question, no additional text.
"""
        
        response = self._timed_api_call(prompt, max_tokens=200)
        return response if response else "Can you provide more specific technical details about your approach?"
    
    def generate_vision_mission_question(self, user_answer, vision_content, jd_content):
        """Generate a vision/mission situational challenge question"""
        
        prompt = f"""Generate a vision/mission situational challenge question based on the user's previous answer:

USER'S PREVIOUS ANSWER: {user_answer}

VISION & MISSION CONTEXT:
Vision: {vision_content[:1500]}
JD: {jd_content[:1000]}

Create a question that:
1. Tests values, decision-making, and alignment with organizational goals
2. Presents a realistic situational challenge
3. Requires vision-driven thinking and leadership
4. Is conversational and professional like a real interview
5. Focuses on culture, values, and strategic alignment
6. Can be answered with examples from work experience

Examples of good vision/mission questions:
- "How would you handle a situation where your team's technical approach conflicts with the company's sustainability goals?"
- "Describe a time when you had to balance innovation with risk management. How did you align this with the company's vision?"

Return only the question, no additional text.
"""
        
        response = self._timed_api_call(prompt, max_tokens=200)
        return response if response else "How do you ensure your technical decisions align with the company's vision and values?"
    
    def generate_leadership_question(self, user_answer, jd_content, vision_content):
        """Generate a cross-functional leadership scenario question"""
        
        prompt = f"""Generate a cross-functional leadership scenario question based on the user's previous answer:

USER'S PREVIOUS ANSWER: {user_answer}

JOB & VISION CONTEXT:
JD: {jd_content[:1500]}
Vision: {vision_content[:1000]}

Create a question that:
1. Tests cross-functional leadership and collaboration skills
2. Presents a realistic team challenge scenario
3. Requires strategic thinking and people management
4. Is conversational and professional like a real interview
5. Focuses on leadership, communication, and conflict resolution
6. Can be answered with examples from work experience

Examples of good leadership questions:
- "Describe a situation where you had to lead a team through a major technical pivot. How did you handle resistance and ensure buy-in?"
- "Tell me about a time when you had to mediate between conflicting technical approaches from different teams. How did you reach consensus?"

Return only the question, no additional text.
"""
        
        response = self._timed_api_call(prompt, max_tokens=200)
        return response if response else "Describe a challenging leadership situation you've faced. How did you handle it and what did you learn?"
    
    def generate_new_technical_question(self, user_answer, jd_content, resume_content):
        """Generate a new technical topic question"""
        
        prompt = f"""Generate a new technical topic question based on the user's previous answer:

USER'S PREVIOUS ANSWER: {user_answer}

JOB & EXPERIENCE CONTEXT:
JD: {jd_content[:1500]}
Resume: {resume_content[:1000]}

Create a question that:
1. Introduces a new technical area or challenge
2. Is relevant to the role and company context
3. Requires problem-solving and technical thinking
4. Is conversational and professional like a real interview
5. Tests adaptability and learning ability
6. Can be answered with technical knowledge and examples

Examples of good new technical questions:
- "How would you approach implementing a real-time data processing system for automotive sensor data?"
- "What's your strategy for ensuring cybersecurity in connected vehicle systems?"

Return only the question, no additional text.
"""
        
        response = self._timed_api_call(prompt, max_tokens=200)
        return response if response else "What's a new technical challenge you'd like to tackle in this role?"
    
    def generate_additional_questions(self, total_questions_asked, jd_content, vision_content, resume_content):
        """Generate additional adaptive questions based on interview progress"""
        
        prompt = f"""Generate 2-3 additional adaptive interview questions based on the interview progress:

INTERVIEW PROGRESS: {total_questions_asked} questions already asked

CONTEXT:
JD: {jd_content[:1000]}
Vision: {vision_content[:800]}
Resume: {resume_content[:1000]}

Create questions that:
1. Build on previous discussion topics
2. Introduce new areas for exploration
3. Are labeled clearly (Technical:, Experience:, Vision/Mission:)
4. Present realistic scenarios and challenges
5. Are conversational and professional like a real interview
6. Test different aspects of the user's capabilities

Format: [Category]: [Specific Topic] - [Detailed scenario-based question]
"""
        
        response = self._timed_api_call(prompt, max_tokens=300)
        if response:
            additional_questions = [q.strip() for q in response.split('\n') if q.strip() and ':' in q]
            return additional_questions
        return []
    
    def evaluate_answer_quality(self, question, answer, jd_content, vision_content, resume_content):
        """Evaluate the quality of the user's answer with multi-parameter assessment"""
        
        # Determine question type for appropriate evaluation criteria
        if 'Technical:' in question:
            evaluation_focus = "Technical Skills and Problem-Solving"
        elif 'Vision/Mission:' in question:
            evaluation_focus = "Vision Alignment and Leadership"
        elif 'Experience:' in question:
            evaluation_focus = "Practical Experience and Decision-Making"
        else:
            evaluation_focus = "Overall Interview Performance"
        
        # Multi-parameter evaluation prompt
        prompt = f"""Evaluate this interview answer with focus on {evaluation_focus}:

QUESTION: {question}
ANSWER: {answer}

CONTEXT:
JD: {jd_content[:1500]}
Vision: {vision_content[:1000]}
Resume: {resume_content[:1500]}

Rate the answer on a scale of 1-10 for each parameter and provide detailed feedback:

EVALUATION PARAMETERS:
1. Technical Skills (1-10): Depth of technical knowledge demonstrated
2. Content Depth (1-10): How comprehensive and detailed the answer is
3. Problem Solving (1-10): Approach to problem-solving and decision-making
4. Communication (1-10): Clarity and structure of explanation
5. Practical Examples (1-10): Use of concrete examples and real-world scenarios

OVERALL SCORE: Average of all parameters

Provide specific feedback for each parameter and overall assessment.
Format: 
Technical Skills: X/10 - [feedback]
Content Depth: X/10 - [feedback]
Problem Solving: X/10 - [feedback]
Communication: X/10 - [feedback]
Practical Examples: X/10 - [feedback]
OVERALL: X/10 - [comprehensive feedback]
"""
        
        response = self._timed_api_call(prompt, max_tokens=400)
        return response if response else "Technical Skills: 7/10 - Good technical understanding shown. Content Depth: 7/10 - Answer covers main points. Problem Solving: 7/10 - Logical approach demonstrated. Communication: 7/10 - Clear explanation. Practical Examples: 6/10 - Could use more specific examples. OVERALL: 7/10 - Solid answer with room for improvement in providing concrete examples."
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'total_api_calls': self._api_calls,
            'total_api_time': self._total_time,
            'avg_api_time': self._total_time / max(self._api_calls, 1)
        }
