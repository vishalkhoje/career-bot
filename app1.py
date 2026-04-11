# User message
#      │
#      ▼
# ┌─────────────┐   ❌ invalid   ┌─────────────────────────┐
# │   Agent 1   │ ─────────────▶ │  OUT_OF_SCOPE_RESPONSE  │
# │ (Guardrail) │               └─────────────────────────┘
# │ lightweight │
# │  model      │   ✅ valid
# └─────────────┘ ─────────────▶ ┌─────────────────────────┐
#                                 │        Agent 2          │
#                                 │  GPT-4.1 + tools        │
#                                 │  (career answer)        │
#                                 └─────────────────────────┘



from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")                     # lightweight model → Agent 1
azure_openai_deployment_premium = os.getenv("AZURE_OPENAI_DEPLOYMENT_PREMIUM", "gpt-4.1")  # GPT-4.1 → Agent 2

OUT_OF_SCOPE_RESPONSE = (
    "I'm designed to assist only with career-related queries based on the provided profile. "
    "Please ask questions related to experience, skills, or projects."
)


# ── Pushover notification ─────────────────────────────────────────────────────
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


# ── Tool functions ────────────────────────────────────────────────────────────
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


# ── Tool schemas (only Agent 2 gets these) ────────────────────────────────────
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name":  {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional context worth recording"},
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

agent2_tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


# ── Main class ────────────────────────────────────────────────────────────────
class Me:

    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT")
        if not azure_openai_api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY")
        if not azure_openai_deployment:
            raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT")

        self.openai = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.azure_endpoint,
            api_key=azure_openai_api_key,
        )

        self.name = "Vishal Khoje"

        linkedin_pdf_path = "me/linkedin.pdf"
        if not os.path.exists(linkedin_pdf_path):
            raise FileNotFoundError("Missing LinkedIn PDF at `me/linkedin.pdf`.")
        reader = PdfReader(linkedin_pdf_path)
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    # =========================================================================
    # AGENT 1 (Validator)— Input Guardrail (lightweight model, classifier only gpt-4.1-mini)
    # =========================================================================

    def agent1_system_prompt(self):
        return """You are a strict input validation agent for a career profile chatbot.

Your ONLY job is to decide whether the user's message is career-related or not.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ VALID messages ask about:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Professional experience, work history, or job roles
- Technical skills, tools, frameworks, or technologies
- Projects, achievements, or certifications
- Education or academic background
- Contact info, GitHub, portfolio, or resume
- Career goals or professional background
- Greetings or introductions (user may be starting a conversation)
- Requests to get in touch or share an email

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ INVALID messages ask about:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- General knowledge (e.g., "What is AI?", "Explain machine learning")
- Personal life unrelated to career (family, relationships, hobbies not in profile)
- News, politics, weather, sports, entertainment
- Jokes, poems, stories, or creative writing
- Any topic clearly unrelated to the person's professional profile

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with valid JSON — no extra text, no markdown fences:
{"is_valid": true, "reason": "brief reason"}
{"is_valid": false, "reason": "brief reason"}"""

    def run_agent1(self, user_message: str) -> tuple[bool, str]:
        """
        Agent 1 (Validator): lightweight guardrail classifier.
        Returns (is_valid: bool, reason: str).
        Uses temperature=0 for deterministic classification.
        """
        print(f"\n[Agent 1 (Validator)] Validating: '{user_message[:80]}'", flush=True)
        try:
            response = self.openai.chat.completions.create(
                model=azure_openai_deployment,      # cheap/fast model is enough here
                messages=[
                    {"role": "system", "content": self.agent1_system_prompt()},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0,   # deterministic — this is a binary classifier
                max_tokens=80,
            )
            raw = response.choices[0].message.content.strip()
            print(f"[Agent 1 (Validator)] Decision: {raw}", flush=True)
            result = json.loads(raw)
            return bool(result.get("is_valid", False)), result.get("reason", "")

        except (json.JSONDecodeError, Exception) as e:
            # Fail safe: if the classifier errors, block the message
            print(f"[Agent 1 (Validator)] Error — blocking by default: {e}", flush=True)
            return False, "Validation agent encountered an error."

    # =========================================================================
    # AGENT 2 — Career Answer Generator (high-configured GPT-4.1)
    # =========================================================================

    def agent2_system_prompt(self):
        return f"""You are a professional AI Career Assistant representing {self.name} on their personal website.
You are provided with two data sources — a LinkedIn profile and a personal summary — \
which contain all the career information you are allowed to use.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Represent {self.name} faithfully, professionally, and with confidence."""

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"[Agent 2] Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    def run_agent2(self, message: str, history: list) -> str:
        """
        Agent 2 (Career Answer Generator) : GPT-4.1 career answer generator with tool-calling loop.
        Only reached if Agent 1 approves the input.
        """
        print(f"[Agent 2] Generating answer with model: {azure_openai_deployment_premium}", flush=True)
        messages = (
            [{"role": "system", "content": self.agent2_system_prompt()}]
            + history
            + [{"role": "user", "content": message}]
        )

        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=azure_openai_deployment_premium,   # GPT-4.1
                messages=messages,
                tools=agent2_tools,
            )

            if response.choices[0].finish_reason == "tool_calls":
                message_obj = response.choices[0].message
                tool_calls = message_obj.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message_obj)
                messages.extend(results)
            else:
                done = True

        return response.choices[0].message.content

    # =========================================================================
    # ORCHESTRATOR — routes Agent 1 → Agent 2
    # =========================================================================

    def chat(self, message: str, history: list) -> str:
        # ── Step 1: Agent 1 validates the input ──
        is_valid, reason = self.run_agent1(message)

        if not is_valid:
            print(f"[Orchestrator] ❌ Blocked — {reason}\n", flush=True)
            return OUT_OF_SCOPE_RESPONSE

        # ── Step 2: Agent 2 generates the answer (GPT-4.1, full capability) ──
        print(f"[Orchestrator] ✅ Approved — {reason}", flush=True)
        return self.run_agent2(message, history)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    me = Me()
    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=me.chat,
            title="Chat with Vishal's AI",
            description=(
                "Skip the standard resume. Ask me directly about Vishal's "
                "technical skills, past projects, and career highlights."
            ),
            # type="messages"
        )
        gr.DeepLinkButton()

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        share=False
    )