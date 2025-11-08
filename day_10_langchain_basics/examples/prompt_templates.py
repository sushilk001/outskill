"""
Prompt Template Examples
Demonstrates various prompt engineering techniques with LangChain.
"""

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def example_basic_template():
    """Example: Basic prompt template"""
    print("\n" + "="*60)
    print("Example 1: Basic Prompt Template")
    print("="*60)
    
    template = """Question: {question}
    
    Answer: Let's think step by step."""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"]
    )
    
    formatted = prompt.format(question="What is the capital of France?")
    print(f"\nFormatted Prompt:\n{formatted}")


def example_chat_template():
    """Example: Chat prompt template with system and human messages"""
    print("\n" + "="*60)
    print("Example 2: Chat Prompt Template")
    print("="*60)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} who speaks in a {style} manner."),
        ("human", "{user_input}"),
    ])
    
    messages = chat_prompt.format_messages(
        role="pirate captain",
        style="dramatic and adventurous",
        user_input="Tell me about your ship"
    )
    
    print("\nFormatted Messages:")
    for msg in messages:
        print(f"{msg.__class__.__name__}: {msg.content}")
    
    # Use with LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.9,
        api_key=Config.OPENAI_API_KEY
    )
    
    response = llm.invoke(messages)
    print(f"\n🏴‍☠️ Response:\n{response.content}")


def example_few_shot_template():
    """Example: Few-shot learning with examples"""
    print("\n" + "="*60)
    print("Example 3: Few-Shot Prompt Template")
    print("="*60)
    
    # Define examples
    examples = [
        {
            "word": "happy",
            "antonym": "sad"
        },
        {
            "word": "tall",
            "antonym": "short"
        },
        {
            "word": "light",
            "antonym": "dark"
        }
    ]
    
    # Create example template
    example_template = """
    Word: {word}
    Antonym: {antonym}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_template
    )
    
    # Create few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of every word\n",
        suffix="\nWord: {input}\nAntonym:",
        input_variables=["input"]
    )
    
    formatted = few_shot_prompt.format(input="brave")
    print(f"\nFormatted Prompt:\n{formatted}")
    
    # Use with LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.3,
        api_key=Config.OPENAI_API_KEY
    )
    
    response = llm.invoke(formatted)
    print(f"\n💡 Response: {response.content}")


def example_structured_output():
    """Example: Prompt for structured output"""
    print("\n" + "="*60)
    print("Example 4: Structured Output Template")
    print("="*60)
    
    template = """Extract the following information from the text:

    Text: {text}
    
    Please provide the information in the following JSON format:
    {{
        "name": "person's name",
        "age": "person's age",
        "occupation": "person's occupation",
        "location": "person's location"
    }}
    
    JSON Output:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"]
    )
    
    text = "John Smith is a 35-year-old software engineer living in San Francisco."
    formatted = prompt.format(text=text)
    
    print(f"\nInput Text: {text}")
    print(f"\nFormatted Prompt:\n{formatted}")
    
    # Use with LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0,
        api_key=Config.OPENAI_API_KEY
    )
    
    response = llm.invoke(formatted)
    print(f"\n📋 Response:\n{response.content}")


def example_conditional_template():
    """Example: Template with conditional logic"""
    print("\n" + "="*60)
    print("Example 5: Conditional Prompt Template")
    print("="*60)
    
    template = """You are a {expertise} expert.

    {% if difficulty == "beginner" %}
    Explain the concept in simple terms suitable for beginners.
    Use analogies and avoid technical jargon.
    {% elif difficulty == "intermediate" %}
    Provide a balanced explanation with some technical details.
    {% else %}
    Give an advanced, technical explanation with implementation details.
    {% endif %}

    Concept: {concept}
    
    Explanation:"""
    
    from langchain.prompts import PromptTemplate
    
    # Note: LangChain's PromptTemplate doesn't support Jinja2 by default
    # This is a conceptual example. For conditional logic, you'd typically:
    # 1. Use Python logic to select different templates
    # 2. Use a custom template engine
    # 3. Build the template programmatically
    
    print("\nThis example demonstrates conditional template structure.")
    print("In practice, you'd select different templates based on conditions.")
    
    # Simple Python-based conditional
    def get_template(difficulty):
        if difficulty == "beginner":
            instruction = "Explain in simple terms suitable for beginners."
        elif difficulty == "intermediate":
            instruction = "Provide a balanced explanation with some technical details."
        else:
            instruction = "Give an advanced, technical explanation."
        
        return f"""You are a {{expertise}} expert.
        
        {instruction}
        
        Concept: {{concept}}
        
        Explanation:"""
    
    prompt = PromptTemplate(
        template=get_template("beginner"),
        input_variables=["expertise", "concept"]
    )
    
    formatted = prompt.format(
        expertise="machine learning",
        concept="neural networks"
    )
    
    print(f"\nFormatted Prompt (Beginner Level):\n{formatted}")


def example_chain_of_thought():
    """Example: Chain-of-thought prompting"""
    print("\n" + "="*60)
    print("Example 6: Chain-of-Thought Template")
    print("="*60)
    
    template = """Solve the following problem step by step:

    Problem: {problem}
    
    Let's approach this systematically:
    
    Step 1: Understand what is being asked
    Step 2: Identify the key information
    Step 3: Apply relevant concepts or formulas
    Step 4: Calculate or reason through to the answer
    Step 5: Verify the answer makes sense
    
    Solution:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["problem"]
    )
    
    problem = "If a train travels at 60 mph for 2.5 hours, how far does it travel?"
    formatted = prompt.format(problem=problem)
    
    print(f"\nProblem: {problem}")
    
    # Use with LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.3,
        api_key=Config.OPENAI_API_KEY
    )
    
    response = llm.invoke(formatted)
    print(f"\n🧠 Chain-of-Thought Solution:\n{response.content}")


def main():
    """Run all examples"""
    if not Config.OPENAI_API_KEY:
        print("\n❌ Error: OPENAI_API_KEY not set!")
        return
    
    print("\n📝 Prompt Template Examples")
    
    try:
        example_basic_template()
        example_chat_template()
        example_few_shot_template()
        example_structured_output()
        example_conditional_template()
        example_chain_of_thought()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

