import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import Tool
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import os

# Set up the page
st.set_page_config(page_title="IntelliSearch AI", page_icon="🔍", layout="wide")
st.title("🔍 IntelliSearch AI - Research Assistant")

# Sidebar for API key configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password", 
                                placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API Key configured!")
    else:
        st.warning("⚠️ Please enter your Groq API key to continue.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    IntelliSearch AI helps you research topics using:
    - **Arxiv** for academic papers
    - **Wikipedia** for general knowledge
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Only proceed if API key is provided
if groq_api_key:
    try:
        # Initialize tools
        arxiv_wrapper = ArxivAPIWrapper()
        wikipedia_wrapper = WikipediaAPIWrapper()

        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

        # Initialize LLM
        llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

        # Create a system prompt that instructs the LLM to use tools
        system_message = SystemMessage(content="""You are a research assistant with access to Arxiv and Wikipedia. 
        When users ask questions, you can search for information using these tools:
        
        - Use Arxiv for academic papers, research articles, and scientific publications
        - Use Wikipedia for general knowledge, definitions, and overviews of topics
        
        Always cite your sources and provide accurate information.""")

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Create chain
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=st.session_state.memory,
            verbose=True
        )

        # Manual tool calling function
        def process_query_with_tools(query):
            """Process query by manually checking if tools are needed and using them"""
            
            # First, let the LLM analyze the query
            analysis = chain.run(input=f"Analyze this query and tell me which tool would be best: {query}. Just respond with 'Arxiv', 'Wikipedia', or 'Neither'.")
            
            tool_response = ""
            
            # Use appropriate tool based on analysis
            if "arxiv" in analysis.lower():
                tool_response = arxiv_tool.run(query)
            elif "wikipedia" in analysis.lower():
                tool_response = wikipedia_tool.run(query)
            else:
                # If no specific tool needed, just respond normally
                return chain.run(input=query)
            
            # Combine tool response with original query for final answer
            final_response = chain.run(input=f"Based on this research: {tool_response}\n\nNow answer the original question: {query}")
            
            return final_response

        # Chat input
        if user_input := st.chat_input("Ask me to research something..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                try:
                    # Show loading indicator
                    with st.spinner("Researching..."):
                        response = process_query_with_tools(user_input)
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    except Exception as e:
        st.error(f"Error initializing tools: {str(e)}")
        st.info("Please check your API key and try again.")
else:
    st.info("👆 Please enter your Groq API key in the sidebar to start using IntelliSearch AI.")
    
    # Example questions
    st.markdown("### Example questions you can ask:")
    st.markdown("""
    - *"Find recent research papers about large language models"*
    - *"Search Wikipedia for information about quantum computing"*
    - *"What are the latest developments in renewable energy according to Arxiv?"*
    - *"Tell me about machine learning from both academic and general knowledge sources"*
    """)
