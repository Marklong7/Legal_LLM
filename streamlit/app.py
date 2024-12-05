import streamlit as st
import vertexai
from google.auth import default
from vertexai.preview.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from llm_agent import evaluate_question
from raptor_script import question_handler
from google_search import generate_google_search_response

# Initialize Vertex AI
PROJECT_ID = "mlds-cap-2024-lexlead-advisor"  # Replace with your project ID
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# Authentication
credentials, project = default()

credentials, project = default()

# Load Gemini models (assuming GenerativeModel works as intended)
try:
    generative_model_gemini_15_pro = GenerativeModel("gemini-1.5-pro-002")
    generative_model_gemini_15_non_pro = GenerativeModel("gemini-1.5-flash-002")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")

# Dictionary of models
models = {
    "LLM Agent Workflow": None,  # Placeholder for LLM Agent Workflow
    "Gemini 1.5 Pro": generative_model_gemini_15_pro,
    "Gemini 1.5 Flash": generative_model_gemini_15_non_pro
}



# Retry logic for generating content
@retry(
    stop=stop_after_attempt(7),  # Increase to 7 attempts
    wait=wait_exponential(multiplier=1, min=5, max=120),  # Min wait 5s, exponential up to 120s
    retry=retry_if_exception_type(ResourceExhausted)
)
def generate_with_retry(model, prompt):
    response = model.generate_content([prompt])
    try:
        # Extract the text from the candidates in the response
        if response and hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0]
        else:
            st.error("No valid response candidates found")
            return "No valid response candidates found."
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        raise

# Streamlit app layout
st.title("Illinois Legal Advisory Chatbot")
st.write("Ask legal questions and get responses from multiple AI models.")

# Initialize conversation state if not already done
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Sidebar for model selection
with st.sidebar:
    st.header("Chat Settings")
    selected_model = st.selectbox("Choose a model:", list(models.keys()))

# User input
question = st.text_input("Ask your legal question:", placeholder="Type your question here...")

# Button to send the message
if st.button("Send"):
    if question and selected_model:
        # Append the user's query to the conversation
        st.session_state.conversation.append({"sender": "You", "text": question})

        try:
            if selected_model == "LLM Agent Workflow":
                # Call evaluate_question for the LLM Agent Workflow
                final_strategy, category, is_rag_useful, difficulty_response = evaluate_question(question)

                # Prepare the reasons and actions as a single string for adding to conversation
                reasoning_text = (
                    f"**Reason-1:** What is the category of the question?\n"
                    f"**Action-1:** {category}\n\n"
                    f"**Reason-2:** Is RAG useful for this task?\n"
                    f"**Action-2:** {'Yes' if is_rag_useful else 'No'}\n\n"
                    f"**Reason-3:** What is the difficulty level of the question?\n"
                    f"**Action-3:** {difficulty_response}\n\n"
                    f"**Final Strategy:** {final_strategy}"
                )

                # Append reasoning to conversation as a system message for reasoning
                st.session_state.conversation.append({"sender": "System Reasoning", "text": reasoning_text})

                # Handle different strategies and generate response
                if "LLM only" in final_strategy or "LLM + self-reflection" in final_strategy:
                    # Use Gemini 1.5 Pro for "LLM only" or "LLM + self-reflection"
                    response = generate_with_retry(generative_model_gemini_15_pro, question)
                    response_text = f"**Final Response:**\n{response}"
                    st.session_state.conversation.append({"sender": "Gemini 1.5 Pro", "text": response_text})

                elif "LLM + Raptor" in final_strategy:
                    # Use Raptor handler "LLM + Raptor"
                    response = question_handler(question)
                    if response:  # Check if response is not None or empty
                        response_text = f"**Final Response:**\n{response}"
                    else:
                        response_text = "**Final Response:**\nNo valid response generated."
                    st.session_state.conversation.append({"sender": "Raptor", "text": response_text})

                # Streamlit app integration
                elif "LLM + Google Search" in final_strategy:
                    # Use Google Search Retrieval to generate content
                    try:
                        search_response = generate_google_search_response(question)
                        response_text = f"**Final Response (via Google Search):** {search_response}"
                    except Exception as e:
                        response_text = f"**Final Response (via Google Search):** Error during Google Search retrieval: {str(e)}"

                    # Append the Google Search response to the conversation
                    st.session_state.conversation.append({"sender": "Google Search", "text": response_text})


            else:
                # Generate response from the selected Gemini model
                model = models[selected_model]
                with st.spinner(f"Generating response with {selected_model}..."):
                    response = generate_with_retry(model, question)
                    response_text = f"**Final Response:**\n{response}"
                    st.session_state.conversation.append({"sender": f"{selected_model}", "text": response_text})

        except Exception as e:
            error_text = f"Error with {selected_model}: {str(e)}"
            st.session_state.conversation.append({"sender": "System", "text": error_text})
    else:
        st.warning("Please enter a question and select a model.")

# Display conversation history (most recent at the top)
st.header("Conversation History")
for message in reversed(st.session_state.conversation):
    if message["sender"] == "You":
        st.markdown(
            f"""
            <div style="color: #007acc; font-weight: bold; margin-bottom: 10px;">
                🧑‍💼 You: {message['text']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif message["sender"] == "System Reasoning":
        # Display reasoning in a different style
        st.markdown(
            f"""
            <div style="color: #666666; font-style: italic; margin-bottom: 10px;">
                🤖 System Reasoning:
                <div style="margin-left: 15px;">{message['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif message["sender"] in ["System", "Gemini 1.5 Pro", "Gemini 1.5 Flash", "LLM Agent Workflow", "RAG", "Raptor", "Final Response"]:
        # Display responses in a standardized style for all system responses
        st.markdown(
            f"""
            <div style="color: #999999; font-weight: bold; margin-bottom: 10px;">
                🤖 {message['sender']} Response:<br>
                <div style="margin-left: 15px;">{message['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Catch-all for any other messages that may not have been handled
        st.markdown(
            f"""
            <div style="color: #999999; font-weight: bold; margin-bottom: 10px;">
                🤖 {message['sender']}:
                <div style="margin-left: 15px;">{message['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
