import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Only initialize agent if API key is provided
if groq_api_key:
    try:
        # Initialize tools
        arxiv_wrapper = ArxivAPIWrapper()
        wikipedia_wrapper = WikipediaAPIWrapper()

        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

        tools = [
            Tool(
                name="Arxiv",
                func=arxiv_tool.run,
                description="Useful for searching academic papers and research articles from Arxiv. Use when you need to find research papers, scientific articles, or academic publications."
            ),
            Tool(
                name="Wikipedia", 
                func=wikipedia_tool.run,
                description="Useful for searching general knowledge and information from Wikipedia. Use when you need factual information, definitions, or overviews of topics."
            )
        ]

        # Initialize LLM
        llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

        # Create ReAct prompt template manually (no langchain-hub needed)
        prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(prompt_template)

        # Create agent
        agent = create_react_agent(llm, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=st.session_state.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        # Chat input
        if user_input := st.chat_input("Ask me to research something..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate agent response
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    response = agent_executor.invoke(
                        {"input": user_input},
                        {"callbacks": [st_callback]}
                    )
                    output = response["output"]
                    st.markdown(output)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": output})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
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
