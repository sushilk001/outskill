"""
Custom Chain Examples
Demonstrates creating custom chains and advanced chain patterns.
"""

from langchain.chains import SequentialChain, LLMChain, TransformChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def example_sequential_chain():
    """Example: Sequential chain with multiple steps"""
    print("\n" + "="*60)
    print("Example: Sequential Chain - Blog Post Generator")
    print("="*60)
    
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.8,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Chain 1: Generate title
    title_prompt = PromptTemplate(
        input_variables=["topic"],
        template="Generate a catchy blog post title about {topic}:\n\nTitle:"
    )
    title_chain = LLMChain(llm=llm, prompt=title_prompt, output_key="title")
    
    # Chain 2: Generate outline
    outline_prompt = PromptTemplate(
        input_variables=["title"],
        template="Create a blog post outline for the title: {title}\n\nOutline:"
    )
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")
    
    # Chain 3: Write introduction
    intro_prompt = PromptTemplate(
        input_variables=["title", "outline"],
        template="""Write an engaging introduction for a blog post.
        
        Title: {title}
        Outline: {outline}
        
        Introduction:"""
    )
    intro_chain = LLMChain(llm=llm, prompt=intro_prompt, output_key="introduction")
    
    # Combine all chains
    overall_chain = SequentialChain(
        chains=[title_chain, outline_chain, intro_chain],
        input_variables=["topic"],
        output_variables=["title", "outline", "introduction"],
        verbose=True
    )
    
    # Run the chain
    topic = "The Future of AI in Education"
    result = overall_chain.invoke({"topic": topic})
    
    print("\n📝 Generated Blog Post:")
    print(f"\n🎯 Title:\n{result['title']}")
    print(f"\n📋 Outline:\n{result['outline']}")
    print(f"\n✍️ Introduction:\n{result['introduction']}")


def example_transform_chain():
    """Example: Transform chain with custom processing"""
    print("\n" + "="*60)
    print("Example: Transform Chain - Text Processing")
    print("="*60)
    
    def transform_func(inputs: dict) -> dict:
        """Custom transformation function"""
        text = inputs["text"]
        # Transform: uppercase and add emoji
        transformed = "🎉 " + text.upper() + " 🎉"
        return {"transformed_text": transformed}
    
    # Create transform chain
    transform_chain = TransformChain(
        input_variables=["text"],
        output_variables=["transformed_text"],
        transform=transform_func
    )
    
    # Use it
    result = transform_chain.invoke({"text": "Hello LangChain!"})
    print(f"\nOriginal: Hello LangChain!")
    print(f"Transformed: {result['transformed_text']}")


def example_router_chain():
    """Example: Router chain for conditional logic"""
    print("\n" + "="*60)
    print("Example: Router-like Chain - Question Classifier")
    print("="*60)
    
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.3,
        api_key=Config.OPENAI_API_KEY
    )
    
    # Classifier chain
    classifier_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Classify the following question into one category:
        - MATH: mathematical or calculation questions
        - CODE: programming or technical questions
        - GENERAL: general knowledge questions
        
        Question: {question}
        
        Category (respond with only one word: MATH, CODE, or GENERAL):"""
    )
    classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt, output_key="category")
    
    # Specialized chains for each category
    math_prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a math expert. Solve this: {question}"
    )
    math_chain = LLMChain(llm=llm, prompt=math_prompt, output_key="answer")
    
    code_prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a programming expert. Answer this: {question}"
    )
    code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="answer")
    
    general_prompt = PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer this: {question}"
    )
    general_chain = LLMChain(llm=llm, prompt=general_prompt, output_key="answer")
    
    # Questions to test
    questions = [
        "What is 15% of 250?",
        "How do I write a list comprehension in Python?",
        "What is the capital of France?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        
        # Classify
        category_result = classifier_chain.invoke({"question": question})
        category = category_result["category"].strip().upper()
        print(f"📂 Category: {category}")
        
        # Route to appropriate chain
        if "MATH" in category:
            chain = math_chain
        elif "CODE" in category:
            chain = code_chain
        else:
            chain = general_chain
        
        # Get answer
        answer_result = chain.invoke({"question": question})
        print(f"💡 Answer: {answer_result['answer']}")


def main():
    """Run all examples"""
    if not Config.OPENAI_API_KEY:
        print("\n❌ Error: OPENAI_API_KEY not set!")
        return
    
    print("\n🔗 Custom Chain Examples")
    
    try:
        example_sequential_chain()
        example_transform_chain()
        example_router_chain()
        
        print("\n✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

