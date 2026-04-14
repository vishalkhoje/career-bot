
#It's a real, valuable tool - the future resume..<br/>
# • Next, improve the resources - add better context about yourself. If you know RAG, then add a knowledge base about you.<br/>
# • Add in more tools! You could have a SQL database with common Q&A that the LLM could read and write from?<br/>
# • Bring in the Evaluator Agent

from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        # Integration with Azure OpenAI
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise ValueError("Missing required environment variable: AZURE_OPENAI_ENDPOINT")
        if not azure_openai_api_key:
            raise ValueError("Missing required environment variable: AZURE_OPENAI_API_KEY")
        if not azure_openai_deployment:
            raise ValueError("Missing required environment variable: AZURE_OPENAI_DEPLOYMENT")

        self.openai = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.azure_endpoint,
            api_key=azure_openai_api_key,
        )

        # Gemini Integration
        # self.GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        # self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        # self.gemini = OpenAI(base_url=self.GEMINI_BASE_URL, api_key=self.GOOGLE_API_KEY)

        self.name = "Vishal Khoje"
        linkedin_pdf_path = "me/linkedin.pdf"
        if not os.path.exists(linkedin_pdf_path):
            raise FileNotFoundError(
                "Missing LinkedIn PDF. Ensure the file exists at `me/linkedin.pdf`."
            )
        reader = PdfReader(linkedin_pdf_path)
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        system_prompt += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 STRICT GUARDRAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ONLY answer questions directly related to {self.name}'s career, experience, skills,
   projects, or professional background as found in the provided documents.

2. Do NOT:
   - Make up or hallucinate any information
   - Answer from general world knowledge
   - Provide assumptions or guesses

3. If the answer is NOT explicitly present in the documents:
   → Respond: "I'm sorry, I can only provide information based on the provided career profile."
   → Then use the record_unknown_question tool to log it.

4. Keep all responses:
   - Professional, confident, and concise
   - Well-structured (use bullet points or sections for multi-part answers)
   - Strictly relevant to the question asked

5. For projects, experience, or skills → highlight role, impact, and technologies used.

6. For links (GitHub, portfolio, etc.) → only share if explicitly present in the documents.

7. If the user seems engaged or interested, invite them to get in touch.
   Ask for their email and record it using the record_user_details tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 CAREER DOCUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Personal Summary:
{self.summary}

## LinkedIn Profile:
{self.linkedin}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            # integration with Azure OpenAI
            response = self.openai.chat.completions.create(
                model=azure_openai_deployment,
                messages=messages,
                tools=tools,
            )

            # integration with GEMINI
            # response = self.gemini.chat.completions.create(model="gemini-2.5-flash", messages=messages, tools=tools)

            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

if __name__ == "__main__":
    me = Me()
    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=me.chat,
            title="Chat with Vishal's AI",
            description="Skip the standard resume. Ask me directly about Vishal's technical skills, past projects, and career highlights.",
            type="messages"
        ) 
        gr.DeepLinkButton()

    demo.queue()  # 🔥 
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,   # 🔥 
        share=False       # 🔥
    )

    