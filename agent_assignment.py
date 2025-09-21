import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to create a vector store from a PDF
def create_vector_store(file_path):
    """Loads a PDF, splits it into chunks, and creates a FAISS vector store."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Function for the summarizer tool
def summarize_text(text: str) -> str:
    """Summarizes a given text using a ChatGroq model."""
    try:
        # We need a separate llm instance for the tool to avoid conflicts
        summarizer_llm = ChatGroq(groq_api_key=st.session_state.api_key, model_name="gemma2-9b-it", temperature=0)
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        response = summarizer_llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error during summarization: {e}"


# --- Streamlit UI Setup ---
st.set_page_config(page_title="IntelliSearch AI", page_icon="🧠")
st.title("🧠 IntelliSearch AI")
st.caption("Your intelligent search and document analysis assistant")

# --- Sidebar Setup ---
st.sidebar.title("Configuration")
st.session_state.api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

st.sidebar.divider()
st.sidebar.subheader("PDF Analysis Tool")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I'm IntelliSearch AI. I can search the web, read arXiv papers, check Wikipedia, analyze your PDFs, and summarize text. How can I assist you today?"
        }
    ]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Main Chat Interface ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- PDF Processing ---
if uploaded_file and not st.session_state.vector_store:
    with st.spinner("Processing PDF... Please wait."):
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create and store the vector store
        st.session_state.vector_store = create_vector_store(uploaded_file.name)
        st.sidebar.success("PDF processed successfully! You can now ask questions about it.")
        # Clean up the temporary file
        os.remove(uploaded_file.name)


# --- User Input and Agent Execution ---
if prompt := st.chat_input("Ask me anything..."):
    if not st.session_state.api_key:
        st.info("Please add your Groq API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(groq_api_key=st.session_state.api_key, model_name="gemma2-9b-it", temperature=0.7, streaming=True)

    # --- Initialize Tools ---
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    
    search_tool = DuckDuckGoSearchRun(name="Web Search")

    summarizer_tool = Tool(
        name="Text Summarizer",
        func=summarize_text,
        description="Useful for summarizing a given block of text. Input should be the text you want to summarize."
    )

    tools = [search_tool, arxiv_tool, wiki_tool, summarizer_tool]

    # Add the PDF QA tool if a vector store exists
    if st.session_state.vector_store:
        pdf_qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=st.session_state.vector_store.as_retriever(),
            chain_type="stuff"
        )
        pdf_tool = Tool(
            name="PDF Question Answering",
            func=pdf_qa_chain.run,
            description="Use this tool to answer questions about the uploaded PDF document. The input should be a clear question about the PDF's content."
        )
        tools.append(pdf_tool)
    
    # --- Agent Initialization ---
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=st.session_state.memory,
        agent_kwargs={"extra_prompt_messages": [chat_history]},
        handle_parsing_errors=True,
        verbose=True
    )

    # --- Get and Display Response ---
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Run the agent with the user's prompt
        response = agent.run(prompt, callbacks=[st_cb])
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
