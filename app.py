import os
import sys

# Change working directory to 2_rag_bot so relative paths and imports resolve correctly
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_BASE_DIR, "2_rag_bot"))
sys.path.insert(0, os.getcwd())

from src.agents.career_agent import CareerAgent
from src.helper.ui import build_ui

agent = CareerAgent()
demo = build_ui(agent)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        share=False,
    )
