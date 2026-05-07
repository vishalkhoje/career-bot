import os
import re

ROOT = "/Users/vishal/projects/personal/career_conversation/2_rag_bot"

REPLACEMENTS = [
    (r"from src\.agent import CareerAgent", r"from src import CareerAgent"),
    (r"from src\.evaluation import EvaluationSystem", r"from src import EvaluationSystem"),
    (r"from src\.monitoring import Observability", r"from src import Observability"),
    (r"from src\.cache import ResponseCache", r"from src import ResponseCache"),
    (r"from src\.retriever import CareerRetriever", r"from src import CareerRetriever"),
    (r"from src import config", r"from src import config"), # This works now
]

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    original = content
    for pattern, repl in REPLACEMENTS:
        content = re.sub(pattern, repl, content)
    
    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print(f"Fixed {path}")

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".py"):
            fix_file(os.path.join(root, file))
