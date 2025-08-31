from pydantic import BaseModel
from openai import OpenAI
import json
from typing import List, Optional

from get_sample_data import sample_jd, sample_resume
from get_prompt import hr_prompt, tech_prompt


class ResponseFormat(BaseModel):
    message: str
    current_question_number: int
    interview_over: bool

# message, history, tech_interview_over, stage (tech_round (default) / hr_round), jd_content, resume_content, hr_policy_doc_content, num_of_ques_tech, num_of_ques_hr

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="AIzaSyBV9OLcYNxZsteHou6CJ0a6HZ-Hb_Ntjbc"
)

model_name = "gemini-2.0-flash" # "gemini-1.5-flash"

jd_content = sample_jd
resume_content = sample_resume
num_of_ques = 3

# message, history, tech_interview_over, stage (tech_round (default) / hr_round), jd_content, resume_content, hr_policy_doc_content, num_of_ques_tech, num_of_ques_hr

def get_chat_response(user_message: str, 
                      history: List[dict], 
                      interview_over: bool,
                      stage: str,
                      jd_content: str,
                      resume_content: str,
                      hr_policy_doc_content: str,
                      num_of_ques_tech: int,
                      num_of_ques_hr: int):


    system_message = ""

    if stage == "tech" and interview_over:
        stage = "hr"

    if stage == "tech":
        system_message = hr_prompt

        system_message = system_message.replace("{num_of_ques}", str(num_of_ques_hr))
        system_message = system_message.replace("{hr_policy_content}", hr_policy_doc_content)
    
    elif stage == "hr":
        system_message = tech_prompt

        system_message = system_message.replace("{num_of_ques}", str(num_of_ques_tech))
        system_message = system_message.replace("{resume_content}", resume_content)
        system_message = system_message.replace("{jd_content}", jd_content)

    else:
        raise Exception("Invalid Stage...!")

    messages_list = [{"role": "system", "content": system_message}]

    if history:
        messages_list += [{"role": h["role"], "content": h["content"]} for h in history]
    messages_list.append({"role": "user", "content": user_message})


    # --- Call Gemini with structured output ---
    response = client.chat.completions.parse(
        model = model_name,
        messages = messages_list,
        response_format=ResponseFormat
    )

    response_content = json.loads(response.choices[0].message.content)

    # print(response_content, type(response_content)) #dict

    return {"response_text": str(response_content)}

re = get_chat_response(user_message="Hi", history=[], interview_over=False, stage="tech", jd_content=sample_jd, resume_content=sample_resume, hr_policy_doc_content="", num_of_ques_hr=3, num_of_ques_tech=3)

print(re)

{'message': "Welcome to the technical interview. Let's start with your experience with Java-based applications. Considering the JD mentions developing high-volume, low-latency applications, can you describe a situation where you had to optimize a Java application for performance, detailing the specific challenges and the techniques you used to overcome them?", 'current_question_number': 1, 'interview_over': False}

# {'response_text': "{'message': 'Hello!', 'current_question_number': 1, 'interview_over': False}"}