import os
import sys
from dotenv import load_dotenv

# Add the directory to path so we can import app
sys.path.append(os.path.join(os.getcwd(), "1_simple_bot"))

from app import Me

def test_simple_bot():
    print("Testing Simple Bot...")
    try:
        me = Me()
        response = me.chat("What are your key skills?", [])
        print(f"Response: {response[:100]}...")
        if len(response) > 0:
            print("✅ Simple Bot logic is working!")
        else:
            print("❌ Simple Bot returned empty response.")
    except Exception as e:
        print(f"❌ Simple Bot failed with error: {e}")

if __name__ == "__main__":
    test_simple_bot()
