# AI_Chat_with_pdf
# 🤖 AI Chat with PDF

<div align="center">

![AI Chat with PDF](https://img.shields.io/badge/AI-Chat%20with%20PDF-blue?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

**Transform your PDF documents into interactive conversations with AI!** 🚀

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

AI Chat with PDF is an intelligent application that allows users to upload PDF documents and engage in natural language conversations about their content. Powered by advanced AI language models and vector embeddings, this tool makes document analysis interactive, efficient, and insightful.

## ✨ Features

### 🔥 Core Functionality
- **📄 PDF Upload & Processing** - Support for multiple PDF formats with automatic text extraction
- **💬 Natural Language Chat** - Ask questions about your documents in plain English
- **🧠 Intelligent Responses** - Get accurate, context-aware answers from your PDFs
- **🔍 Smart Search** - Advanced semantic search across document content
- **📊 Document Analysis** - Summarize, extract insights, and analyze document themes

### 🚀 Advanced Features
- **⚡ Real-time Processing** - Instant responses with optimized performance
- **🎯 Contextual Understanding** - Maintains conversation context for follow-up questions
- **📚 Multi-document Support** - Chat with multiple PDFs simultaneously
- **💾 Session Memory** - Remembers conversation history within sessions
- **🎨 User-friendly Interface** - Clean, intuitive web interface built with Streamlit

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **🤖 AI/ML** | ![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white) ![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=flat&logoColor=black) |
| **🖥️ Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) |
| **⚙️ Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) |
| **🗄️ Vector DB** | ![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=flat&logo=pinecone&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat&logoColor=white) ![FAISS](https://img.shields.io/badge/FAISS-4B8BBE?style=flat&logoColor=white) |
| **📄 PDF Processing** | ![PyPDF2](https://img.shields.io/badge/PyPDF2-FF4444?style=flat&logoColor=white) ![PDFplumber](https://img.shields.io/badge/PDFplumber-336791?style=flat&logoColor=white) |

</div>

## 🚀 Installation

### Prerequisites
- 🐍 Python 3.8 or higher
- 🔑 OpenAI API key (or other supported AI provider)
- 💾 Git

### Quick Start

1. **📥 Clone the repository**
   ```bash
   git clone https://github.com/Harshjais12/AI_Chat_with_pdf.git
   cd AI_Chat_with_pdf
   ```

2. **🔧 Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **📦 Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **🔐 Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **🎬 Run the application**
   ```bash
   streamlit run app.py
   ```

6. **🌐 Open your browser** and navigate to `http://localhost:8501`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# AI Provider Configuration
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here

# Vector Database (choose one)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# Application Settings
MAX_FILE_SIZE=50  # MB
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📝 Usage

### Basic Workflow

1. **📤 Upload PDF**: Click the upload button and select your PDF document
2. **⏳ Processing**: Wait for the document to be processed and indexed
3. **💭 Ask Questions**: Type your questions about the document content
4. **💡 Get Answers**: Receive intelligent, context-aware responses
5. **🔄 Continue Conversation**: Ask follow-up questions for deeper insights

### Example Queries

```
🔍 "What is the main topic of this document?"
📊 "Can you summarize the key findings?"
🎯 "What does the author say about [specific topic]?"
📈 "Show me the statistics mentioned in chapter 3"
🔍 "Find all references to [keyword]"
```

## 🏗️ Project Structure

```
AI_Chat_with_pdf/
├── 📁 app/
│   ├── 🐍 main.py                 # Main Streamlit application
│   ├── 🔧 pdf_processor.py        # PDF processing utilities
│   ├── 🤖 chat_engine.py          # AI chat functionality
│   └── 💾 vector_store.py         # Vector database operations
├── 📁 config/
│   └── ⚙️ settings.py             # Configuration settings
├── 📁 utils/
│   ├── 🔨 helpers.py              # Utility functions
│   └── 🎨 ui_components.py        # Custom UI components
├── 📁 static/
│   ├── 🎨 styles.css              # Custom styles
│   └── 🖼️ images/                 # Image assets
├── 📄 requirements.txt            # Dependencies
├── 🔧 .env.example               # Environment template
├── 📖 README.md                  # This file
└── 🚀 app.py                     # Application entry point
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🔧 Development Setup

1. **🍴 Fork the repository**
2. **🌿 Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **💻 Make your changes**
4. **✅ Test thoroughly**
5. **📝 Commit with descriptive messages**
   ```bash
   git commit -m "✨ Add amazing feature"
   ```
6. **📤 Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **🎯 Open a Pull Request**

### 🐛 Bug Reports

Found a bug? Please open an issue with:
- 📋 Clear description of the problem
- 🔄 Steps to reproduce
- 💻 Your environment details
- 📸 Screenshots if applicable

## 🗺️ Roadmap

- [ ] 🌐 **Multi-language Support** - Support for PDFs in different languages
- [ ] 📱 **Mobile App** - Native mobile application
- [ ] 🔒 **Authentication** - User accounts and document management
- [ ] 📊 **Analytics Dashboard** - Usage statistics and insights
- [ ] 🎙️ **Voice Input** - Ask questions using voice commands
- [ ] 📤 **Export Features** - Export conversations and summaries
- [ ] 🔌 **API Integration** - RESTful API for developers
- [ ] 🎨 **Custom Themes** - Personalized UI themes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🤖 **OpenAI** for providing powerful language models
- 🦜 **LangChain** for the excellent framework
- 🎈 **Streamlit** for the amazing web framework
- 🌟 **Open Source Community** for inspiration and support

## 📞 Support

Having issues? We're here to help!

- 📧 **Email**: support@example.com
- 💬 **Discord**: [Join our community](https://discord.gg/example)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Harshjais12/AI_Chat_with_pdf/issues)
- 📖 **Documentation**: [Wiki](https://github.com/Harshjais12/AI_Chat_with_pdf/wiki)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Harsh Jain](https://github.com/Harshjais12)

[🔝 Back to top](#-ai-chat-with-pdf)

</div>
