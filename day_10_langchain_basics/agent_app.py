"""
LangChain Agent Application
Demonstrates agents with custom tools for autonomous task execution.
"""

import os
from datetime import datetime
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from config import Config


# Define custom tools

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current date and time in the specified format."""
    return datetime.now().strftime(format)


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any math calculations.
    Example: '2 + 2' or '10 * 5 + 3'
    """
    try:
        # Safety: Only allow basic math operations
        allowed_chars = set('0123456789+-*/()%. ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, %) allowed."
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words in the given text."""
    words = text.split()
    return f"The text contains {len(words)} words."


@tool
def text_reverser(text: str) -> str:
    """Reverse the given text."""
    return text[::-1]


@tool
def create_file(filename: str, content: str) -> str:
    """
    Create a file with the given filename and content in the data directory.
    Returns a success or error message.
    """
    try:
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        filepath = os.path.join(Config.DATA_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return f"✅ File created successfully: {filepath}"
    except Exception as e:
        return f"❌ Error creating file: {str(e)}"


@tool
def read_file(filename: str) -> str:
    """
    Read and return the contents of a file from the data directory.
    Provide just the filename, not the full path.
    """
    try:
        filepath = os.path.join(Config.DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            return f"❌ File not found: {filename}"
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        return f"File contents:\n{content}"
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


def create_agent():
    """Create a ReAct agent with custom tools"""
    print("\n🤖 Creating agent with tools...")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0,  # Lower temperature for more deterministic agent behavior
        api_key=Config.OPENAI_API_KEY
    )
    
    # Define tools
    tools = [
        get_current_time,
        calculate,
        word_counter,
        text_reverser,
        create_file,
        read_file,
    ]
    
    # Create prompt template for ReAct agent
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
    )
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=Config.MAX_ITERATIONS,
        handle_parsing_errors=True
    )
    
    print(f"✅ Agent created with {len(tools)} tools")
    return agent_executor


def run_agent_examples(agent_executor):
    """Run example agent tasks"""
    print("\n" + "="*60)
    print("🚀 Running Agent Examples")
    print("="*60)
    
    examples = [
        "What is the current date and time?",
        "Calculate 15% tip on a $85 restaurant bill",
        "Count how many words are in this sentence: The quick brown fox jumps over the lazy dog",
        "Reverse the text 'Hello World'",
        "Create a file called 'test.txt' with the content 'This is a test file created by an AI agent'",
        "Read the file 'test.txt'",
    ]
    
    for i, task in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {task}")
        print('='*60)
        
        try:
            result = agent_executor.invoke({"input": task})
            print(f"\n✅ Result: {result['output']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


def interactive_agent(agent_executor):
    """Interactive agent session"""
    print("\n" + "="*60)
    print("💬 Interactive Agent Mode (type 'quit' to exit)")
    print("="*60)
    print("\nThe agent can help you with:")
    print("  • Time/date queries")
    print("  • Math calculations")
    print("  • Text operations (word count, reverse)")
    print("  • File operations (create, read)")
    print("\nJust describe what you want in natural language!\n")
    
    while True:
        task = input("Your task: ").strip()
        
        if task.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! 👋")
            break
        
        if not task:
            continue
        
        try:
            result = agent_executor.invoke({"input": task})
            print(f"\n✅ Result: {result['output']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main function"""
    if not Config.OPENAI_API_KEY:
        print("\n❌ Error: OPENAI_API_KEY not set!")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    try:
        # Create agent
        agent_executor = create_agent()
        
        # Run examples
        run_agent_examples(agent_executor)
        
        # Ask if user wants interactive mode
        print("\n" + "="*60)
        choice = input("\nWould you like to try interactive agent mode? (yes/no): ").strip().lower()
        
        if choice in ['yes', 'y']:
            interactive_agent(agent_executor)
        
        print("\n✅ Agent demo completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

