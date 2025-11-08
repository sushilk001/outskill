"""
Conversation Application with Memory
Demonstrates different types of conversation memory in LangChain.
"""

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import Config


def create_llm():
    """Create and return an LLM instance"""
    return ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.TEMPERATURE,
        api_key=Config.OPENAI_API_KEY
    )


def example_buffer_memory():
    """Example: Conversation with buffer memory (keeps all history)"""
    print("\n" + "="*60)
    print("Example 1: Buffer Memory (Keeps All History)")
    print("="*60)
    
    llm = create_llm()
    memory = ConversationBufferMemory()
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Have a conversation
    print("\n--- Conversation Start ---")
    
    response1 = conversation.predict(input="Hi! My name is Alice and I love Python programming.")
    print(f"\nAssistant: {response1}")
    
    response2 = conversation.predict(input="What's my name?")
    print(f"\nAssistant: {response2}")
    
    response3 = conversation.predict(input="What programming language do I like?")
    print(f"\nAssistant: {response3}")
    
    # Show memory
    print("\n--- Memory Contents ---")
    print(memory.load_memory_variables({}))


def example_window_memory():
    """Example: Conversation with window memory (keeps last N messages)"""
    print("\n" + "="*60)
    print("Example 2: Window Memory (Keeps Last N Messages)")
    print("="*60)
    
    llm = create_llm()
    memory = ConversationBufferWindowMemory(k=2)  # Keep only last 2 exchanges
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print("\n--- Conversation Start (Window size: 2) ---")
    
    response1 = conversation.predict(input="My favorite color is blue.")
    print(f"\nAssistant: {response1}")
    
    response2 = conversation.predict(input="I work as a data scientist.")
    print(f"\nAssistant: {response2}")
    
    response3 = conversation.predict(input="I have two cats.")
    print(f"\nAssistant: {response3}")
    
    # This should not remember the color (too old)
    response4 = conversation.predict(input="What's my favorite color?")
    print(f"\nAssistant: {response4}")
    
    print("\n--- Memory Contents (only last 2 exchanges) ---")
    print(memory.load_memory_variables({}))


def example_summary_memory():
    """Example: Conversation with summary memory (summarizes old messages)"""
    print("\n" + "="*60)
    print("Example 3: Summary Memory (Summarizes History)")
    print("="*60)
    
    llm = create_llm()
    memory = ConversationSummaryMemory(llm=llm)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print("\n--- Conversation Start ---")
    
    response1 = conversation.predict(
        input="I'm planning a trip to Japan next month. I'm excited about visiting Tokyo and Kyoto."
    )
    print(f"\nAssistant: {response1}")
    
    response2 = conversation.predict(
        input="I love sushi and ramen. Can't wait to try authentic Japanese food!"
    )
    print(f"\nAssistant: {response2}")
    
    response3 = conversation.predict(
        input="Where am I planning to travel?"
    )
    print(f"\nAssistant: {response3}")
    
    print("\n--- Summary Memory Contents ---")
    print(memory.load_memory_variables({}))


def custom_conversation_with_context():
    """Example: Custom conversation with specific context"""
    print("\n" + "="*60)
    print("Example 4: Custom Conversation with Context")
    print("="*60)
    
    llm = create_llm()
    
    # Custom prompt template
    template = """You are a friendly Python programming tutor. You help students learn Python
    by providing clear explanations and encouraging them.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    memory = ConversationBufferMemory()
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    
    print("\n--- Python Tutor Conversation ---")
    
    response1 = conversation.predict(input="How do I create a list in Python?")
    print(f"\nTutor: {response1}")
    
    response2 = conversation.predict(input="Can you show me how to add items to it?")
    print(f"\nTutor: {response2}")
    
    response3 = conversation.predict(input="Great! What about removing items?")
    print(f"\nTutor: {response3}")


def interactive_chat():
    """Interactive chat session"""
    print("\n" + "="*60)
    print("Interactive Chat (type 'quit' to exit)")
    print("="*60)
    
    llm = create_llm()
    memory = ConversationBufferMemory()
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    print("\nChat started! Type your messages below.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        try:
            response = conversation.predict(input=user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Run all examples"""
    if not Config.OPENAI_API_KEY:
        print("\n❌ Error: OPENAI_API_KEY not set!")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    print("\n💬 LangChain Conversation Memory Examples")
    print("This script demonstrates different types of conversation memory.\n")
    
    try:
        # Run automated examples
        example_buffer_memory()
        example_window_memory()
        example_summary_memory()
        custom_conversation_with_context()
        
        # Ask if user wants interactive mode
        print("\n" + "="*60)
        choice = input("\nWould you like to try interactive chat? (yes/no): ").strip().lower()
        
        if choice in ['yes', 'y']:
            interactive_chat()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

