# main.py
import sys
from app.chatbot.chatbot import SecuredChatbot
from app.auth.rbac import RBAC

def main():
    print("=" * 60)
    print("   GENERIC SECURE CHATBOT WORKSPACE: INITIATED ")
    print("=" * 60)
    
    print("\nAvailable user profiles for this session:")
    for uid, account in RBAC.USER_DIRECTORY.items():
        print(f"  [{uid}] - Role: {account['role']}")
    print("-" * 60)
    
    user_id = input("\nSelect User Account ID string to launch session context: ").strip()
    if user_id not in RBAC.USER_DIRECTORY:
        print("Invalid operator sequence assigned. Program shutting down.")
        sys.exit(1)
        
    print(f"\nAuthorization validated. Assigned Role: {RBAC.USER_DIRECTORY[user_id]['role']}")
    print("System active. Type 'exit' to cleanly close your channels.\n")
    
    # Intitialize master orchestrator code instance
    engine = SecuredChatbot()
    
    while True:
        try:
            user_prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not user_prompt:
            continue
        if user_prompt.lower() in ["exit", "quit"]:
            break
            
        ai_reply = engine.process_message(user_id=user_id, message=user_prompt)
        print(f"AI: {ai_reply}\n")

if __name__ == "__main__":
    main()