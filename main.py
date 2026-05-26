# main.py
import sys
from app.chatbot.chatbot import SecuredChatbot  # Fixed path!
from app.auth.rbac import RBAC                   # Fixed path!
from app.rag.secure_rag import SecureRAGPipeline
def main():
    print("=" * 60)
    print("   SECURE WORKSPACE SYSTEM MODULE — ACTIVE")
    print("=" * 60)
    
    print("\nProfiles available:")
    for uid, data in RBAC.USER_DIRECTORY.items():
        print(f"  [{uid}] Role: {data['role']}")
        
    user_id = input("\nEnter User Account ID string context: ").strip()
    if user_id not in RBAC.USER_DIRECTORY:
        print("Invalid operational sequence.")
        sys.exit(1)

    print("\nSelect Mode:")
    print("  [1] Chatbot (Week 1)")
    print("  [2] Secure Production RAG (Week 2)")
    mode = input("Choice: ").strip()

    if mode == "2":
        print("\n--- Running Secure RAG Mode (Type 'exit' to quit) ---")
        pipeline = SecureRAGPipeline()
        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue
            res = pipeline.execute_query(user_id, query)
            print(f"AI: {res['response']}\n")
    else:
        print("\n--- Running General Chatbot Mode (Type 'exit' to quit) ---")
        engine = SecuredChatbot()
        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]: break
            if not query: continue
            print(f"AI: {engine.process_message(user_id, query)}\n")

if __name__ == "__main__":
    main()