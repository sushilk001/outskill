# 🚀 Streamlit Chatbot Interface

A beautiful and feature-rich chatbot interface built with Streamlit, featuring customizable settings, session statistics, and chat export functionality.

## ✨ Features

- **Interactive Chat Interface**: Modern chat UI with message history
- **Customizable Assistant Settings**:
  - Custom assistant name
  - Multiple response styles (Friendly, Professional, Casual, Technical, Creative)
- **Chat Configuration**:
  - Adjustable maximum chat history (10-100 messages)
  - Toggle timestamps on/off
- **Session Statistics**:
  - Real-time session duration tracking
  - Message count tracking (user messages and total messages)
- **Action Buttons**:
  - Clear chat history
  - Export chat as downloadable .txt file
- **Responsive Design**: Clean, modern UI with custom styling
- **Typing Effect**: Simulated typing animation for assistant responses

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🔧 Installation

1. **Clone or download this repository**

2. **Navigate to the project directory**:
```bash
cd Day_3
```

3. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

Or install Streamlit directly:
```bash
pip install streamlit>=1.28.0
```

## 🚀 Running the Application

1. **Start the Streamlit app**:
```bash
streamlit run chatbot_app.py
```

2. **Access the application**:
   - The app will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`

## 📖 Usage Guide

### Sidebar Configuration

#### Assistant Settings
- **Assistant Name**: Customize the name of your chatbot assistant
- **Response Style**: Choose from 5 different response styles:
  - Friendly: Warm and welcoming responses
  - Professional: Formal and business-appropriate responses
  - Casual: Relaxed and informal responses
  - Technical: Detailed technical responses
  - Creative: Imaginative and innovative responses

#### Chat Settings
- **Max Chat History**: Set the maximum number of messages to keep (10-100)
  - Helps manage memory and display
  - Older messages are automatically trimmed
- **Show Timestamps**: Toggle to show/hide message timestamps

### Session Stats
Real-time statistics displayed in the sidebar:
- **Session Duration**: Time elapsed since session started
- **Messages Sent**: Number of messages you've sent
- **Total Messages**: Total messages in the conversation (user + assistant)

### Actions
- **🗑️ Clear Chat**: Removes all messages and resets statistics
- **💾 Export**: Downloads chat history as a .txt file
  - Includes assistant name, response style, and timestamps
  - File named with current date and time

### Chat Interface
1. Type your message in the input box at the bottom
2. Press Enter or click the send button
3. Watch the assistant respond with a typing effect
4. Scroll through chat history as needed

### Expandable Sections
- **📘 About This Demo**: Information about features and capabilities
- **📝 Instructor Notes**: Technical implementation details and future enhancements

## 🎨 UI Components

The interface includes:
- **Main Chat Area**: Displays conversation with user and assistant messages
- **Sidebar**: Configuration panel and statistics
- **Message Bubbles**: Distinct styling for user and assistant messages
- **Avatars**: Visual indicators (👤 for user, 🤖 for assistant)
- **Timestamps**: Optional time stamps for each message

## 🛠️ Technical Details

### Built With
- **Streamlit**: Web framework for the interface
- **Python**: Backend logic and session management
- **CSS**: Custom styling for enhanced UI

### Session State Management
The app uses Streamlit's session state to persist:
- Chat message history
- User preferences (assistant name, response style)
- Session statistics
- Configuration settings

### File Structure
```
Day_3/
├── chatbot_app.py       # Main application file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔮 Future Enhancements

Potential improvements for the chatbot:
- Integration with AI/LLM APIs (OpenAI GPT, Claude, etc.)
- Database persistence for chat history
- User authentication and multi-user support
- Multi-language support
- Voice input/output capabilities
- File upload and processing
- Conversation branching and context management
- Advanced analytics and insights

## 🐛 Troubleshooting

### Port Already in Use
If port 8501 is already in use, run with a different port:
```bash
streamlit run chatbot_app.py --server.port 8502
```

### Dependencies Issues
Make sure you're using Python 3.7+:
```bash
python --version
```

Update pip before installing dependencies:
```bash
pip install --upgrade pip
```

### Browser Not Opening
If the browser doesn't open automatically, manually navigate to:
```
http://localhost:8501
```

## 📝 Notes

- The chatbot currently provides simulated responses based on the selected style
- For production use, integrate with actual AI/LLM APIs
- Session data is not persisted between browser sessions
- Chat export includes all visible messages in the current history

## 📄 License

This project is created for educational purposes.

## 👥 Support

For issues or questions, please refer to the [Streamlit documentation](https://docs.streamlit.io).

---

**Enjoy your chatbot experience! 🎉**

