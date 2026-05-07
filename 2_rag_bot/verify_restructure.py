import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv(override=True)

try:
    from src.agents.career_agent import CareerAgent
    print("✅ CareerAgent import successful.")
    
    agent = CareerAgent()
    print("✅ CareerAgent initialization successful.")
    
    # Test a simple chat turn (mocking history)
    # We use next() because chat is a generator
    gen = agent.chat("Hi, who are you?", [])
    first_response = next(gen)
    print(f"✅ CareerAgent chat start: {first_response}")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
