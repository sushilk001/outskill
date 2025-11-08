"""
Basic LangChain Examples
Demonstrates fundamental LangChain concepts with simple examples.
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import ChatOpenAI
from config import Config


def example_1_simple_llm():
    """Example 1: Basic LLM call"""
    print("\n" + "="*60)
    print("Example 1: Simple LLM Call")
    print("="*60)
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.TEMPERATURE,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Simple invoke
    response = llm.invoke("What is LangChain? Answer in one sentence.")
    print(f"\nResponse: {response.content}")


def example_2_prompt_template():
    """Example 2: Using Prompt Templates"""
    print("\n" + "="*60)
    print("Example 2: Prompt Templates")
    print("="*60)
    
    # Create a prompt template
    template = """You are a helpful assistant that explains concepts simply.
    
    Concept: {concept}
    Audience: {audience}
    
    Please explain the concept in a way that the audience will understand:"""
    
    prompt = PromptTemplate(
        input_variables=["concept", "audience"],
        template=template
    )
    
    # Format the prompt
    formatted_prompt = prompt.format(
        concept="Machine Learning",
        audience="a 10-year-old child"
    )
    
    print("\nFormatted Prompt:")
    print(formatted_prompt)
    
    # Use with LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.7,
        api_key=Config.OPENAI_API_KEY
    )
    
    response = llm.invoke(formatted_prompt)
    print(f"\nResponse:\n{response.content}")


def example_3_llm_chain():
    """Example 3: LLMChain - Combining LLM with Prompt"""
    print("\n" + "="*60)
    print("Example 3: LLMChain")
    print("="*60)
    
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.8,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["topic", "style"],
        template="Write a {style} story about {topic}. Keep it under 100 words."
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    result = chain.invoke({
        "topic": "a robot learning to paint",
        "style": "humorous"
    })
    
    print(f"\nStory:\n{result['text']}")


def example_4_sequential_chain():
    """Example 4: Sequential Chain - Multiple steps"""
    print("\n" + "="*60)
    print("Example 4: Sequential Chain")
    print("="*60)
    
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.7,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Chain 1: Generate a product name
    prompt1 = PromptTemplate(
        input_variables=["product_description"],
        template="Create a catchy product name for: {product_description}\n\nProduct Name:"
    )
    chain1 = LLMChain(llm=llm, prompt=prompt1)
    
    # Chain 2: Generate a tagline for the product
    prompt2 = PromptTemplate(
        input_variables=["product_name"],
        template="Create a memorable tagline for a product named: {product_name}\n\nTagline:"
    )
    chain2 = LLMChain(llm=llm, prompt=prompt2)
    
    # Combine into sequential chain
    overall_chain = SimpleSequentialChain(
        chains=[chain1, chain2],
        verbose=True
    )
    
    # Run the chain
    product_description = "An AI-powered app that helps people learn languages through conversations"
    result = overall_chain.invoke(product_description)
    
    print(f"\nFinal Result: {result['output']}")


def example_5_chat_prompt():
    """Example 5: Chat Prompt Templates"""
    print("\n" + "="*60)
    print("Example 5: Chat Prompt Templates")
    print("="*60)
    
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.7,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Create chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding assistant who explains concepts clearly."),
        ("human", "Explain {concept} in Python with a simple example."),
    ])
    
    # Create chain
    chain = chat_prompt | llm
    
    # Run
    result = chain.invoke({"concept": "list comprehension"})
    print(f"\nResponse:\n{result.content}")


def main():
    """Run all examples"""
    if not Config.OPENAI_API_KEY:
        print("\n❌ Error: OPENAI_API_KEY not set!")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    print("\n🚀 LangChain Basic Examples")
    print("This script demonstrates fundamental LangChain concepts.\n")
    
    try:
        example_1_simple_llm()
        example_2_prompt_template()
        example_3_llm_chain()
        example_4_sequential_chain()
        example_5_chat_prompt()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your OpenAI API key is valid and you have credits.")


if __name__ == "__main__":
    main()

