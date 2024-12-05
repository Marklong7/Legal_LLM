import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding
from google.auth import default
import streamlit as st
import subprocess

PROJECT_ID = "mlds-cap-2024-lexlead-advisor" 
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# Authentication
credentials, project = default()

# Load Generative Model (e.g., Gemini)
try:
    generative_model = GenerativeModel("gemini-1.5-pro-002")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()

# Function to call Google Search script
def call_google_search_script():
    try:
        result = subprocess.run(['python3', 'google_search.py'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return result.stdout
        else:
            print(f"Error executing google_search.py: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception occurred while calling google_search.py: {str(e)}")
        return None

# Function to clean the grounding response
def clean_grounding_response(response):
    # Remove extra lines and spaces for better readability
    if isinstance(response, str):
        cleaned_response = " ".join([line.strip() for line in response.splitlines() if line.strip()])
        return cleaned_response
    else:
        return str(response)

# Function to generate response using Google Search Retrieval Tool
def generate_google_search_response(prompt):
    # Create Google Search Retrieval Tool
    try:
        tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
    except AttributeError as e:
        print(f"Error creating Google Search Retrieval Tool: {str(e)}")
        return None

    # Generate response using the tool
    try:
        response = generative_model.generate_content(prompt, tools=[tool])
        if hasattr(response, 'candidates'):
            for candidate in response.candidates:
                cleaned_content = clean_grounding_response(candidate.content.parts[0])
                return cleaned_content
        else:
            print("No valid response candidates found.")
            return "No valid response candidates found."
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        return None

