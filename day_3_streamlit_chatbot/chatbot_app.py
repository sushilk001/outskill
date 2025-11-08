import streamlit as st
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Cool Assistant Chatbot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #2b2b2b;
    }
    .user-message {
        background-color: #1e1e1e;
        border-left: 3px solid #ff4b4b;
    }
    .assistant-message {
        background-color: #2b2b2b;
        border-left: 3px solid #ffa500;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0.3rem;
    }
    .stat-box {
        padding: 0.5rem;
        background-color: #1e1e1e;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
    
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
    
if 'user_message_count' not in st.session_state:
    st.session_state.user_message_count = 0

if 'assistant_name' not in st.session_state:
    st.session_state.assistant_name = "This is a cool assistant!"
    
if 'response_style' not in st.session_state:
    st.session_state.response_style = "Friendly"
    
if 'max_history' not in st.session_state:
    st.session_state.max_history = 40
    
if 'show_timestamps' not in st.session_state:
    st.session_state.show_timestamps = True

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Assistant Settings
    with st.expander("**Assistant Settings**", expanded=True):
        st.markdown("**Assistant Name:**")
        assistant_name = st.text_input(
            "Assistant Name",
            value=st.session_state.assistant_name,
            label_visibility="collapsed"
        )
        st.session_state.assistant_name = assistant_name
        
        st.markdown("**Response Style:**")
        response_style = st.selectbox(
            "Response Style",
            ["Friendly", "Professional", "Casual", "Technical", "Creative"],
            index=["Friendly", "Professional", "Casual", "Technical", "Creative"].index(st.session_state.response_style),
            label_visibility="collapsed"
        )
        st.session_state.response_style = response_style
    
    # Chat Settings
    with st.expander("**Chat Settings**", expanded=True):
        st.markdown("**Max Chat History:**")
        max_history = st.slider(
            "Max Chat History",
            min_value=10,
            max_value=100,
            value=st.session_state.max_history,
            label_visibility="collapsed"
        )
        st.session_state.max_history = max_history
        
        show_timestamps = st.checkbox(
            "Show Timestamps",
            value=st.session_state.show_timestamps
        )
        st.session_state.show_timestamps = show_timestamps
    
    st.markdown("---")
    
    # Session Stats
    st.markdown("## 📊 Session Stats")
    
    # Calculate session duration
    duration = datetime.now() - st.session_state.start_time
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    
    st.markdown(f"""
    <div class="stat-box">
        <strong>Session Duration</strong><br>
        {duration_str}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <strong>Messages Sent</strong><br>
        {st.session_state.user_message_count}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-box">
        <strong>Total Messages</strong><br>
        {st.session_state.message_count}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action Buttons
    st.markdown("## 🎯 Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.session_state.user_message_count = 0
            st.session_state.start_time = datetime.now()
            st.rerun()
    
    with col2:
        if st.button("💾 Export"):
            if st.session_state.messages:
                # Prepare chat export
                export_text = f"Chat Export - {st.session_state.assistant_name}\n"
                export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                export_text += f"Response Style: {st.session_state.response_style}\n"
                export_text += "="*50 + "\n\n"
                
                for msg in st.session_state.messages:
                    role = "You" if msg["role"] == "user" else st.session_state.assistant_name
                    timestamp = msg.get("timestamp", "")
                    export_text += f"[{timestamp}] {role}:\n{msg['content']}\n\n"
                
                # Create download button
                st.download_button(
                    label="Download Chat",
                    data=export_text,
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.info("No messages to export")

# Main chat interface
st.markdown(f'<div class="main-header">🚀 {st.session_state.assistant_name}</div>', unsafe_allow_html=True)

st.markdown(f"**Response Style:** {st.session_state.response_style} | **History Limit:** {st.session_state.max_history} messages")

st.markdown("---")

# Display initial greeting if no messages
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.write("Hello! I'm your demo assistant. How can I help you today?")

# Display chat messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        if st.session_state.show_timestamps and "timestamp" in message:
            st.markdown(f'<div class="timestamp">{message["role"].capitalize()} - {message["timestamp"]}</div>', unsafe_allow_html=True)
        st.write(message["content"])

# Chat input
if prompt := st.chat_input(f"Message {st.session_state.assistant_name}..."):
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    st.session_state.message_count += 1
    st.session_state.user_message_count += 1
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        if st.session_state.show_timestamps:
            st.markdown(f'<div class="timestamp">You - {timestamp}</div>', unsafe_allow_html=True)
        st.write(prompt)
    
    # Generate assistant response based on response style
    with st.chat_message("assistant", avatar="🤖"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if st.session_state.show_timestamps:
            st.markdown(f'<div class="timestamp">{st.session_state.assistant_name} - {timestamp}</div>', unsafe_allow_html=True)
        
        # Simulate different response styles
        response_templates = {
            "Friendly": f"Hey, great question about '{prompt}'! I'm happy to help you with that. Here's what I'm thinking...",
            "Professional": f"Thank you for your inquiry. Regarding '{prompt}', I can provide the following information...",
            "Casual": f"Cool! So you're asking about '{prompt}'. Let me break it down for you...",
            "Technical": f"Query received: '{prompt}'. Processing response with technical specifications...",
            "Creative": f"Ooh, interesting! '{prompt}' - that sparks some creative ideas! Let me share my thoughts..."
        }
        
        response = response_templates.get(st.session_state.response_style, response_templates["Friendly"])
        
        # Simulate typing effect
        message_placeholder = st.empty()
        full_response = ""
        
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.write(full_response + "▌")
        
        message_placeholder.write(full_response)
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "timestamp": timestamp
    })
    st.session_state.message_count += 1
    
    # Trim messages if exceeds max history
    if len(st.session_state.messages) > st.session_state.max_history:
        st.session_state.messages = st.session_state.messages[-st.session_state.max_history:]
    
    st.rerun()

# Expandable sections at the bottom
st.markdown("---")

with st.expander("📘 About This Demo"):
    st.markdown("""
    This is a demonstration chatbot interface built with Streamlit. It features:
    
    - **Customizable Assistant**: Change the assistant name and response style
    - **Chat History Management**: Set maximum chat history length
    - **Session Statistics**: Track session duration and message counts
    - **Export Functionality**: Download your chat history as a text file
    - **Timestamp Display**: Toggle timestamps on/off for messages
    - **Responsive Design**: Modern UI with smooth interactions
    
    The chatbot simulates responses based on the selected response style (Friendly, Professional, Casual, Technical, or Creative).
    """)

with st.expander("📝 Instructor Notes"):
    st.markdown("""
    ### Implementation Details:
    
    **Technologies Used:**
    - Streamlit >= 1.28.0
    - Python 3.7+
    
    **Key Features Implemented:**
    1. **Session State Management**: Persistent state across reruns
    2. **Real-time Stats**: Dynamic session duration and message counting
    3. **Export Functionality**: Download chat as .txt file
    4. **Custom Styling**: CSS for enhanced UI/UX
    5. **Configurable Settings**: Assistant name, response style, history limit
    
    **Architecture:**
    - Uses Streamlit's session state for data persistence
    - Implements chat_message components for message display
    - Sidebar for configuration and stats
    - Main area for chat interface
    
    **Future Enhancements:**
    - Integration with actual AI/LLM APIs (OpenAI, Anthropic, etc.)
    - Database persistence for chat history
    - User authentication
    - Multi-language support
    - Voice input/output
    """)

# Auto-refresh for session duration (updates every 10 seconds)
st.markdown("""
<script>
    setTimeout(function() {
        window.parent.location.reload();
    }, 10000);
</script>
""", unsafe_allow_html=True)

