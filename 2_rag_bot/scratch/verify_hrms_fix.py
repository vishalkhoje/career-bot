import sys
import os

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))

from src import CareerAgent

def verify_hrms():
    print("--- 🔍 HRMS Context Continuity Verification ---")
    agent = CareerAgent()
    
    query = "in your project details vishal share more information about HRMS & Payroll SaaS details with organisation and duration"
    
    print(f"Querying: {query}\n")
    
    full_response = ""
    for chunk in agent.chat(query, []):
        full_response += chunk
        
    print("\n--- BOT RESPONSE ---")
    print(full_response)
    print("\n--- END OF RESPONSE ---")
    
    # Check for keywords that indicate success
    if "sumHR" in full_response:
        print("\n✅ SUCCESS: Bot provided details and correctly identified the organization as sumHR!")
    elif "not provide" in full_response.lower() or "do not have access" in full_response.lower():
        print("\n❌ FAILED: Bot still says details are missing.")
    else:
        print("\n⚠️ UNCERTAIN: Bot responded but 'sumHR' was not found. Please review manually.")

if __name__ == "__main__":
    verify_hrms()
