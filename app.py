import os
import sys
import gradio as gr

# Change working directory to 2_rag_bot so relative paths and imports resolve correctly
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_BASE_DIR, "2_rag_bot"))
sys.path.insert(0, os.getcwd())

from src.agents.career_agent import CareerAgent
from src.helper.ui import build_ui
from dashboard import build_dashboard

agent = CareerAgent()

# Warm up the retrieval pipeline in background so the first query is fast
import threading
threading.Thread(target=agent.retriever.warmup, daemon=True).start()

# Build the separate Gradio Blocks
chat_ui = build_ui(agent)
dashboard_ui = build_dashboard()

# Merge them under a single gr.Blocks with Tabs
with gr.Blocks(title="Vishal's AI Career Assistant & Analytics", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("💬 Career Conversation"):
            chat_ui.render()
        with gr.Tab("📊 Analytics Dashboard"):
            dashboard_ui.render()

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        share=False,
    )
