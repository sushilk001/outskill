"""
Streamlit Web UI for LangChain Application
Interactive web interface for all LangChain features.
"""

import streamlit as st
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import Config
import os


# Page configuration
st.set_page_config(
    page_title="LangChain Interactive App",
    page_icon="🦜",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()


def check_api_key():
    """Check if API key is configured"""
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("api_key", "")
    return bool(api_key)


def get_llm(api_key=None):
    """Get LLM instance"""
    key = api_key or Config.OPENAI_API_KEY or st.session_state.get("api_key")
    return ChatOpenAI(
        model=st.session_state.get("model", Config.DEFAULT_MODEL),
        temperature=st.session_state.get("temperature", Config.TEMPERATURE),
        api_key=key
    )


def sidebar_config():
    """Sidebar configuration"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("api_key", Config.OPENAI_API_KEY),
            help="Enter your OpenAI API key"
        )
        st.session_state.api_key = api_key
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
        st.session_state.model = model
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random"
        )
        st.session_state.temperature = temperature
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory()
            st.rerun()


def simple_chat_tab():
    """Simple chat interface"""
    st.header("💬 Simple Chat")
    st.write("Chat with an AI assistant")
    
    if not check_api_key():
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm()
                    conversation = ConversationChain(
                        llm=llm,
                        memory=st.session_state.memory,
                        verbose=False
                    )
                    response = conversation.predict(input=prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")


def prompt_playground_tab():
    """Prompt template playground"""
    st.header("🎨 Prompt Playground")
    st.write("Experiment with prompt templates")
    
    if not check_api_key():
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prompt Template")
        
        template = st.text_area(
            "Template",
            value="You are a {role}. {task}",
            height=150,
            help="Use {variable_name} for variables"
        )
        
        # Extract variables from template
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        
        st.subheader("Variables")
        variable_values = {}
        for var in variables:
            variable_values[var] = st.text_input(f"{var}", key=f"var_{var}")
    
    with col2:
        st.subheader("Output")
        
        if st.button("▶️ Generate", type="primary"):
            if all(variable_values.values()):
                try:
                    llm = get_llm()
                    prompt = PromptTemplate(
                        input_variables=list(variable_values.keys()),
                        template=template
                    )
                    chain = LLMChain(llm=llm, prompt=prompt)
                    
                    with st.spinner("Generating..."):
                        result = chain.invoke(variable_values)
                        st.markdown("### Result")
                        st.write(result['text'])
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please fill in all variables")


def text_analysis_tab():
    """Text analysis tools"""
    st.header("📝 Text Analysis")
    st.write("Analyze and transform text")
    
    if not check_api_key():
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
        return
    
    text_input = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Paste your text here..."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Summarize"):
            if text_input:
                with st.spinner("Summarizing..."):
                    try:
                        llm = get_llm()
                        response = llm.invoke(f"Summarize this text concisely:\n\n{text_input}")
                        st.success("Summary:")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with col2:
        if st.button("🎯 Extract Key Points"):
            if text_input:
                with st.spinner("Extracting..."):
                    try:
                        llm = get_llm()
                        response = llm.invoke(f"Extract the key points from this text as a bullet list:\n\n{text_input}")
                        st.success("Key Points:")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with col3:
        if st.button("🔄 Rephrase"):
            if text_input:
                with st.spinner("Rephrasing..."):
                    try:
                        llm = get_llm()
                        response = llm.invoke(f"Rephrase this text in a different way:\n\n{text_input}")
                        st.success("Rephrased:")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")


def about_tab():
    """About page"""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## LangChain Interactive App
    
    This is an interactive web interface for exploring LangChain capabilities.
    
    ### Features:
    - **Simple Chat**: Conversational AI with memory
    - **Prompt Playground**: Experiment with prompt templates
    - **Text Analysis**: Summarize, extract key points, and rephrase text
    
    ### Tech Stack:
    - LangChain
    - OpenAI GPT models
    - Streamlit
    
    ### Setup:
    1. Enter your OpenAI API key in the sidebar
    2. Select your preferred model and temperature
    3. Start exploring!
    
    ### Resources:
    - [LangChain Documentation](https://python.langchain.com/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [OpenAI API](https://platform.openai.com/)
    """)


def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🦜🔗 LangChain Interactive App")
    
    # Sidebar configuration
    sidebar_config()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Simple Chat",
        "🎨 Prompt Playground",
        "📝 Text Analysis",
        "ℹ️ About"
    ])
    
    with tab1:
        simple_chat_tab()
    
    with tab2:
        prompt_playground_tab()
    
    with tab3:
        text_analysis_tab()
    
    with tab4:
        about_tab()


if __name__ == "__main__":
    main()

