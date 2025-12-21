import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

# Set up HuggingFace token
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize search tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)



st.title("🔎 Combined PDF RAG & AI Search Chatbot")
st.write("Upload PDFs to chat with their content OR use AI-powered web search")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Mode selection
mode = st.sidebar.selectbox(
    "Choose Mode:",
    ("PDF Chat", "AI Search", "Combined Mode")
)

session_id = st.sidebar.text_input("Session ID", value="default_session")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can chat with PDFs and search the web. How can I help you?"}
    ]

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    
    # PDF Upload section (only show in PDF or Combined mode)
    if mode in ["PDF Chat", "Combined Mode"]:
        st.subheader("📄 PDF Upload")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        # Initialize variables to avoid UnboundLocalError
        if 'rag_chain' not in st.session_state:
            st.session_state.rag_chain = None
        if 'get_session_history' not in st.session_state:
            st.session_state.get_session_history = None
        
        if uploaded_files:
            with st.spinner("Processing PDF files..."):
                documents = []
                processed_files = []
                
                for uploaded_file in uploaded_files:
                    try:
                        # Show progress for each file
                        st.info(f"Processing {uploaded_file.name}...")
                        
                        temppdf = f"./temp_{uploaded_file.name}"
                        with open(temppdf, "wb") as file:
                            file.write(uploaded_file.getvalue())
                        
                        loader = PyPDFLoader(temppdf)
                        docs = loader.load()
                        
                        # Filter out empty documents and add file info
                        valid_docs = []
                        for doc in docs:
                            if doc.page_content.strip():
                                # Add source file information to metadata
                                doc.metadata['source_file'] = uploaded_file.name
                                valid_docs.append(doc)
                        
                        if valid_docs:
                            documents.extend(valid_docs)
                            processed_files.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name}: {len(valid_docs)} pages extracted")
                        else:
                            st.warning(f"⚠️ {uploaded_file.name}: No readable text found")
                        
                        # Clean up temp file
                        if os.path.exists(temppdf):
                            os.remove(temppdf)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
                        # Clean up temp file even if there's an error
                        if 'temppdf' in locals() and os.path.exists(temppdf):
                            os.remove(temppdf)
                        continue
                
                # Check if we have any valid documents
                if not documents:
                    st.error("❌ No valid content found in any of the uploaded PDF files.")
                    st.info("💡 Please ensure your PDFs contain readable text (not just images) and try again.")
                    # Clear any existing RAG chain
                    st.session_state.rag_chain = None
                    st.session_state.get_session_history = None
                else:
                    try:
                        st.info(f"📝 Found {len(documents)} pages of content. Creating knowledge base...")
                        
                        # Create vector store using FAISS instead of Chroma
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,  # Reduced chunk size for better retrieval
                            chunk_overlap=200,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        # Check if we have any splits after text splitting
                        if not splits:
                            st.error("❌ No text content could be extracted and split from the PDF files.")
                            st.session_state.rag_chain = None
                            st.session_state.get_session_history = None
                        else:
                            st.info(f"🔄 Creating embeddings for {len(splits)} text chunks...")
                            
                            # Create embeddings with progress indication
                            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                            retriever = vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 3}  # Return top 3 most relevant chunks
                            )
                            
                            st.info("🤖 Setting up conversational AI...")
                            
                            # Set up RAG chain (moved inside this try block)
                            contextualize_q_system_prompt = (
                                "Given a chat history and the latest user question "
                                "which might reference context in the chat history, "
                                "formulate a standalone question which can be understood "
                                "without the chat history. Do NOT answer the question, "
                                "just reformulate it if needed and otherwise return it as is."
                            )
                            
                            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                                ("system", contextualize_q_system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ])
                            
                            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                            
                            system_prompt = (
                                "You are an assistant for question-answering tasks. "
                                "Use the following pieces of retrieved context to answer "
                                "the question. If you don't know the answer based on the "
                                "provided context, say that you don't know. Keep your answers "
                                "concise but informative."
                                "\n\n"
                                "{context}"
                            )
                            
                            qa_prompt = ChatPromptTemplate.from_messages([
                                ("system", system_prompt),
                                MessagesPlaceholder("chat_history"),
                                ("human", "{input}"),
                            ])
                            
                            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                            
                            def get_session_history(session: str) -> BaseChatMessageHistory:
                                if session_id not in st.session_state.store:
                                    st.session_state.store[session_id] = ChatMessageHistory()
                                return st.session_state.store[session_id]
                            
                            conversational_rag_chain = RunnableWithMessageHistory(
                                rag_chain, get_session_history,
                                input_messages_key="input",
                                history_messages_key="chat_history",
                                output_messages_key="answer"
                            )
                            
                            # Store in session state
                            st.session_state.rag_chain = conversational_rag_chain
                            st.session_state.get_session_history = get_session_history
                            st.session_state.processed_files = processed_files
                            st.session_state.total_chunks = len(splits)
                            
                            st.success(f"🎉 Successfully processed {len(processed_files)} PDF(s)!")
                            st.info(f"📊 Knowledge base created with {len(splits)} text chunks from {len(documents)} pages")
                            
                            # Show processed files
                            with st.expander("📁 Processed Files"):
                                for file in processed_files:
                                    st.write(f"✅ {file}")
                            
                    except Exception as e:
                        st.error(f"❌ Error creating knowledge base: {str(e)}")
                        st.error("💡 This might be due to:")
                        st.write("- Empty or corrupted PDF content")
                        st.write("- Very large PDF files (try smaller files)")
                        st.write("- Network issues with embedding model")
                        st.session_state.rag_chain = None
                        st.session_state.get_session_history = None
        
        # Show current status
        if hasattr(st.session_state, 'rag_chain') and st.session_state.rag_chain is not None:
            st.success("✅ PDF knowledge base is ready! You can now ask questions about your documents.")
            if hasattr(st.session_state, 'processed_files'):
                st.caption(f"Loaded files: {', '.join(st.session_state.processed_files)}")
        else:
            st.info("📤 Please upload PDF files to start chatting with your documents.")

    
    # Set up search agent (for AI Search and Combined mode)
    if mode in ["AI Search", "Combined Mode"]:
        # Initialize search agent if not exists
        if 'search_agent' not in st.session_state:
            st.session_state.search_agent = None
            
        tools = [arxiv, wiki]
        search_agent = initialize_agent(
            tools, llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            handling_parser_errors=True
        )
        st.session_state.search_agent = search_agent
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            if mode == "PDF Chat":
                # Handle PDF chat only
                if hasattr(st.session_state, 'rag_chain') and st.session_state.rag_chain is not None:
                    try:
                        session_history = st.session_state.get_session_history(session_id)
                        response = st.session_state.rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}},
                        )
                        answer = response['answer']
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error processing your question: {str(e)}")
                        st.write("Please try rephrasing your question or upload the PDF files again.")
                else:
                    st.write("Please upload PDF files first to use PDF chat mode.")
            
            elif mode == "AI Search":
                # Handle AI search only
                if hasattr(st.session_state, 'search_agent') and st.session_state.search_agent is not None:
                    try:
                        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        response = st.session_state.search_agent.run(prompt, callbacks=[st_cb])
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        st.write("Please try a different search query.")
                else:
                    st.write("Search agent not initialized. Please check your API key.")
            
            elif mode == "Combined Mode":
                # Handle combined mode - try PDF first, then search if needed
                pdf_response = None
                
                if hasattr(st.session_state, 'rag_chain') and st.session_state.rag_chain is not None:
                    try:
                        session_history = st.session_state.get_session_history(session_id)
                        rag_response = st.session_state.rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}},
                        )
                        pdf_response = rag_response['answer']
                        
                        # Check if the PDF response indicates it doesn't know the answer
                        if "don't know" in pdf_response.lower() or "cannot answer" in pdf_response.lower():
                            st.write("📄 PDF Response:")
                            st.write(pdf_response)
                            st.write("\n🔍 Searching the web for additional information...")
                            
                            # Use search agent for additional information
                            if hasattr(st.session_state, 'search_agent') and st.session_state.search_agent is not None:
                                try:
                                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                                    search_response = st.session_state.search_agent.run(prompt, callbacks=[st_cb])
                                    final_response = f"PDF Response: {pdf_response}\n\nWeb Search Results: {search_response}"
                                    st.write("\n📊 Combined Response:")
                                    st.write(search_response)
                                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                                except Exception as e:
                                    st.error(f"Search error: {str(e)}")
                                    st.session_state.messages.append({"role": "assistant", "content": pdf_response})
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": pdf_response})
                        else:
                            st.write("📄 PDF Response:")
                            st.write(pdf_response)
                            st.session_state.messages.append({"role": "assistant", "content": pdf_response})
                    
                    except Exception as e:
                        st.error(f"Error with PDF processing: {str(e)}")
                        # Fall back to search
                        if hasattr(st.session_state, 'search_agent') and st.session_state.search_agent is not None:
                            try:
                                st.write("🔍 Falling back to web search...")
                                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                                response = st.session_state.search_agent.run(prompt, callbacks=[st_cb])
                                st.write(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            except Exception as search_e:
                                st.error(f"Search also failed: {str(search_e)}")
                                st.write("Both PDF and search failed. Please check your setup and try again.")
                        else:
                            st.write("PDF processing failed and search agent not available.")
                
                else:
                    # No PDF loaded, use search only
                    if hasattr(st.session_state, 'search_agent') and st.session_state.search_agent is not None:
                        try:
                            st.write("🔍 No PDFs loaded, using web search...")
                            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                            response = st.session_state.search_agent.run(prompt, callbacks=[st_cb])
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Search error: {str(e)}")
                            st.write("Please try a different search query.")
                    else:
                        st.write("Please upload PDF files or ensure search agent is initialized.")

else:
    st.warning("Please enter your Groq API Key in the sidebar to get started.")

# Display session information in sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can chat with PDFs and search the web. How can I help you?"}
    ]
    if 'store' in st.session_state:
        st.session_state.store.clear()
    st.rerun()

# Show current session info
if st.sidebar.checkbox("Show Session Info"):
    st.sidebar.write(f"Current Session: {session_id}")
    st.sidebar.write(f"Messages in session: {len(st.session_state.messages)}")
    if 'store' in st.session_state and session_id in st.session_state.store:
        st.sidebar.write(f"Chat history length: {len(st.session_state.store[session_id].messages)}")


