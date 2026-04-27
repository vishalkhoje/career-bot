
#It's a real, valuable tool - the future resume..<br/>
# • Next, improve the resources - add better context about yourself. If you know RAG, then add a knowledge base about you.<br/>
# • Add in more tools! You could have a SQL database with common Q&A that the LLM could read and write from?<br/>
# • Bring in the Evaluator Agent

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import time
import sqlite3
import hashlib
from pypdf import PdfReader
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_community.callbacks.manager import get_openai_callback
from pinecone import Pinecone


load_dotenv(override=True)
azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


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
        # Integration with OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Missing required environment variable: OPENAI_API_KEY")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_path = os.path.join(current_dir, ".langchain.db")
        self.last_cache_check = time.time()
        self._init_manual_cache()
        
        self.openai = OpenAI(
            api_key=self.openai_api_key,
        )
        
        # LangChain Chat Model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.openai_api_key,
            temperature=0,
            cache=False # Disable LangChain's internal cache to use our manual one
        ).bind_tools(tools)

        # Gemini Integration
        # self.GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        # self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        # self.gemini = OpenAI(base_url=self.GEMINI_BASE_URL, api_key=self.GOOGLE_API_KEY)

        self.name = "Vishal Khoje"
        
        # Pinecone Vector Store initialization
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "career-bot")
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", # Standard OpenAI embedding model
            api_key=self.openai_api_key,
        )
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key
        )


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
    
    def system_prompt(self, context):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given relevant snippets from {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "
        system_prompt += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 STRICT GUARDRAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ONLY answer questions directly related to {self.name}'s career, experience, skills,
   projects, or professional background as found in the provided context.

2. Do NOT:
   - Make up or hallucinate any information
   - Answer from general world knowledge
   - Provide assumptions or guesses

3. If the answer is NOT explicitly present in the context:
   → Respond: "I'm sorry, I can only provide information based on the provided career profile."
   → Then use the record_unknown_question tool to log it.

4. Keep all responses:
   - Professional, confident, and concise
   - Well-structured (use bullet points or sections for multi-part answers)
   - Strictly relevant to the question asked

5. For projects, experience, or skills → highlight role, impact, and technologies used.

6. For links (GitHub, portfolio, etc.) → only share if explicitly present in the context.
   - If the user asks for a resume, explicitly look for the "Resume Download link" in the context and provide it.
   - You ARE allowed to share links found in the context.

7. If the user seems engaged or interested, invite them to get in touch.
   Ask for their email and record it using the record_user_details tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 CAREER CONTEXT (RELEVANT SNIPPETS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def _init_manual_cache(self):
        """Initialize a manual SQLite table for response caching."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                query_hash TEXT PRIMARY KEY,
                response_text TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()

    def _get_manual_cache(self, query_hash):
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute('SELECT response_text FROM responses WHERE query_hash = ?', (query_hash,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def _set_manual_cache(self, query_hash, response_text):
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO responses (query_hash, response_text, timestamp) VALUES (?, ?, ?)',
                      (query_hash, response_text, time.time()))
        conn.commit()
        conn.close()

    def _generate_cache_key(self, messages):
        """Create a unique hash for the full message chain."""
        # Convert messages to a stable JSON string for hashing
        msg_data = []
        for m in messages:
            msg_dict = {
                "type": m.type,
                "content": m.content.strip() if isinstance(m.content, str) else m.content,
            }
            # Include tool calls in hash if present to distinguish between different agent actions
            if hasattr(m, "tool_calls"):
                msg_dict["tool_calls"] = getattr(m, "tool_calls", [])
            msg_data.append(msg_dict)
        
        stable_str = json.dumps(msg_data, sort_keys=True)
        query_hash = hashlib.sha256(stable_str.encode()).hexdigest()
        
        # DEBUG: Help user understand why cache hits or misses
        print(f"\n--- Cache Debug ---")
        print(f"Query Hash: {query_hash}")
        # print(f"Messages being hashed: {stable_str[:200]}...") 
        print(f"-------------------\n")
        
        return query_hash
    def _generate_cache_key_simple(self, message_list):
        """Create a simple hash for cache keys from message list."""
        stable_str = json.dumps(message_list, sort_keys=True)
        return hashlib.sha256(stable_str.encode()).hexdigest()

    def _check_cache_expiry(self):
        """Internal helper to clear cache if older than 24 hours."""
        if os.path.exists(self.cache_path):
            file_age = time.time() - os.path.getmtime(self.cache_path)
            if file_age > 86400: # 24 hours in seconds
                print(f"Cache expired (Age: {file_age:.0f}s). Clearing for fresh response...")
                try:
                    os.remove(self.cache_path)
                    self._init_manual_cache()
                except Exception as e:
                    print(f"Warning: Failed to clear expired cache: {e}")
        self.last_cache_check = time.time()

    def chat(self, message, history):
        # 0. Normalize message for stable cache keys
        message = message.strip()

        # 0. Check for cache expiration
        self._check_cache_expiry()

        # ✅ 1. Cache key = normalized message ONLY (history-independent).
        #    Career answers are factual — same question always gets same answer.
        #    This ensures cache hits whether user asks in turn 1, 2, or 10.
        normalized_message = message.lower().strip()
        cache_key = hashlib.sha256(normalized_message.encode()).hexdigest()
        print(f"\n[Cache] key={cache_key[:12]}  query='{message}'")

        cached_res = self._get_manual_cache(cache_key)
        if cached_res:
            print(f"⚡ [CACHE HIT] Served from SQLite. Cost: $0.00")
            return cached_res

        print(f"🔍 [CACHE MISS] Calling LLM...")

        # 2. Retrieve relevant context from Pinecone (only on cache miss)
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(message, k=10)
                docs.sort(key=lambda x: x.page_content)
                context = "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"Error during similarity search: {e}")
                context = "Context retrieval failed."
        else:
            context = "Context retrieval is currently disabled (missing configuration)."

        # 3. Build LangChain messages
        langchain_messages = [SystemMessage(content=self.system_prompt(context))]
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                langchain_messages.append(HumanMessage(content=item[0]))
                langchain_messages.append(AIMessage(content=item[1]))
        langchain_messages.append(HumanMessage(content=message))

        # 3. Execute with Token Tracking and Tracing (Only if cache miss)
        start_time = time.time()
        with get_openai_callback() as cb:
            done = False
            while not done:
                # Using LangChain's LLM
                response = self.llm.invoke(langchain_messages)
                
                if response.tool_calls:
                    langchain_messages.append(response)
                    results = self.handle_tool_call_langchain(response.tool_calls)
                    # Convert tool results to ToolMessage objects
                    for res in results:
                        langchain_messages.append(ToolMessage(
                            content=res["content"],
                            tool_call_id=res["tool_call_id"]
                        ))
                else:
                    done = True
            
            # Store in manual cache for next time
            self._set_manual_cache(cache_key, response.content)
            
            duration = time.time() - start_time
            print(f"\n--- Token Usage & Cost (CACHE MISS) ---")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost:.6f}")
            print(f"Response Time: {duration:.2f}s")
            print(f"---------------------------\n")
        
        return response.content

    def handle_tool_call_langchain(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            arguments = tool_call['args']
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call['id']
            })
        return results

if __name__ == "__main__":
    me = Me()
    
    # Environment-based configuration
    env = os.getenv("ENV", "production")
    chat_kwargs = {
        "fn": me.chat,
        "title": "Chat with Vishal's AI",
        "description": "Skip the standard resume. Ask me directly about Vishal's technical skills, past projects, and career highlights.",
    }
    
    # Gradio 5.x uses type="messages", Gradio 4.x (local/dev) uses tuples
    if env == "production":
        chat_kwargs["type"] = "messages"
        print("Running in PRODUCTION mode with Gradio 5 compatibility.")
    else:
        print("Running in DEVELOPMENT mode with Gradio 4 compatibility.")

    with gr.Blocks() as demo:
        gr.ChatInterface(**chat_kwargs) 
        gr.DeepLinkButton()

    demo.queue()  # 🔥 
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,   # 🔥 
        share=False       # 🔥
    )

    