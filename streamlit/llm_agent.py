import vertexai
from google.auth import default
from vertexai.preview.generative_models import GenerativeModel
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Initialize Vertex AI
PROJECT_ID = "mlds-cap-2024-lexlead-advisor"  # Replace with your project ID
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# Authentication
credentials, project = default()

# Load the Gemini model
generative_model_gemini_15_pro = GenerativeModel("gemini-1.5-pro-002")

@retry(
    stop=stop_after_attempt(7),  # Retry up to 7 attempts
    wait=wait_exponential(multiplier=1, min=5, max=120),  # Min wait 5s, exponential up to 120s
    retry=retry_if_exception_type(ResourceExhausted)
)
def generate_response(model, prompt):
    try:
        response = model.generate_content([prompt])
        response_dict = response.to_dict()

        if 'candidates' in response_dict and response_dict['candidates']:
            candidate = response_dict['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                text_response = candidate['content']['parts'][0].get('text', '').strip()
                return text_response if text_response else "Error: No text found in response."
            else:
                return "Error: No valid content or parts found in candidate."
        else:
            return "Error: No valid candidates found in response."
    except Exception as e:
        raise

# Function to determine if RAG (Retrieval-Augmented Generation) is useful based on the category
def should_use_rag_with_gemini(question_or_category):
    prompt = f"""
    Based on the following question or category, determine if RAG (Retrieval-Augmented Generation) is useful. 
    If the question requires recalling external legal information or citation, RAG is useful.
    If the question is focused on interpretation or understanding without external references, RAG is not useful.
    
    Question/Category: {question_or_category}
    
    Answer with 'Yes' or 'No' based on whether RAG is useful.
    """
    response = generate_response(generative_model_gemini_15_pro, prompt)
    print(f"Reason-2: Is RAG useful for this task?\nAnswer-2: {'Yes' if 'yes' in response.lower() else 'No'}")
    return "yes" in response.lower()

# Function to determine if Raptor is needed for high-level information
def should_use_raptor(question):
    prompt = f"""
    Determine if the following question requires high-level information that is best provided by Raptor, or if regular RAG is sufficient.
    
    Examples:
    1. Question: "Provide a summary of the legal landscape for environmental regulations in Illinois."
       Answer: "Raptor"
    2. Question: "What are the key requirements of the Illinois Clean Air Act?"
       Answer: "RAG"
    3. Question: "Explain the impact of Illinois privacy laws on businesses operating in multiple states."
       Answer: "Raptor"
    4. Question: "List the penalties for non-compliance with the Illinois Consumer Fraud Act."
       Answer: "RAG"
    
    Question: {question}
    
    Answer with 'Raptor' or 'RAG' based on the need for high-level information.
    """
    response = generate_response(generative_model_gemini_15_pro, prompt)
    print(f"Reason-5: Does the question need high-level information?\nAnswer-5: {response.strip()}")
    return response.strip()

# Function to classify the question into one of the predefined categories
def classify_question_category(question):
    prompt = f"""
    Goal: Classify the following question into one of the categories: issue-spotting, rule-recall, rule-application, rule-conclusion, interpretation, rhetorical-understanding, out-of-scope.
    
    Examples:
    1. Question: "What is the main legal issue in the case of Roe v. Wade?"
       Answer: "issue-spotting" (Identifying the central issue in a case)
    2. Question: "What are the legal requirements for establishing negligence in a personal injury case?"
       Answer: "rule-recall" (Asking for specific legal rules related to negligence)
    3. Question: "How would the rules of negligence apply if a defendant causes a car accident while texting?"
       Answer: "rule-application" (Applying the rules of negligence to a specific scenario)
    4. Question: "If a person is found to have committed fraud, what legal consequences can they face under state law?"
       Answer: "rule-conclusion" (Drawing conclusions about the legal consequences of fraud)
    5. Question: "How should the term 'reasonable efforts' be interpreted in a contract clause?"
       Answer: "interpretation" (Interpreting the meaning of the term 'reasonable efforts' in a contract)
    6. Question: "What are the rhetorical strategies used in the court's ruling on same-sex marriage?"
       Answer: "rhetorical-understanding" (Analyzing the rhetorical strategies in the ruling)
    7. Question: "What is the capital of France?"
       Answer: "out-of-scope" (Unrelated to the legal domain)

    Question: {question}
    
    Provide the most appropriate category for the question.
    """
    response = generate_response(generative_model_gemini_15_pro, prompt)
    print(f"Reason-1: What is the category of the question?\nAnswer-1: {response.strip()}")
    return response.strip() if "Error" not in response else response

# Function to classify the difficulty of the question based on category and other factors
def classify_difficulty(question, category):
    category_performance = {
        "rule-recall": 30,
        "rule-application": 50,
        "rule-conclusion": 40,
        "interpretation": 80,
        "rhetorical-understanding": 70,
        "issue-spotting": 60,
    }

    category_difficulty = "Hard"
    if category in category_performance:
        performance = category_performance[category]
        if performance >= 70:
            category_difficulty = "Easy"
        elif performance >= 50:
            category_difficulty = "Medium"

    prompt = f"""
    Goal: Determine the difficulty of the following question based on its category and classification along two dimensions:
    1. Explicit vs Implicit: Is the question asking for direct information (explicit) or requiring inference and deeper understanding (implicit)?
    2. Local vs Summary: Does the question focus on a specific sentence or span (local) or ask for the overall meaning of the text (summary)?

    Consider the following guidelines for each category based on performance:
    - **Rule-recall**: Hard (Low accuracy from models like GPT-4: ~30%).
    - **Rule-application**: Medium (Moderate accuracy from models like GPT-4: ~50%).
    - **Rule-conclusion**: Hard (Low accuracy from models like GPT-4: ~40%).
    - **Interpretation**: Easy (High accuracy from models like GPT-4: ~80%).
    - **Rhetorical-understanding**: Medium (Moderate accuracy from models like GPT-4: ~70%).
    - **Issue-spotting**: Medium (Moderate accuracy from models like GPT-4: ~60%).

    If the category provides clear guidance on difficulty, use it. If not, classify based on explicit/implicit and local/summary.

    Question: "{question}"
    Category: {category}

    Based on the question and category, classify the difficulty as **Easy**, **Medium**, or **Hard**. Also, classify the question as **Explicit** or **Implicit**, and **Local** or **Summary**.
    """
    response = generate_response(generative_model_gemini_15_pro, prompt)
    print(f"Reason-3: What is the difficulty level of the question?\nAnswer-3: {response.strip()}")
    return response.strip() if "Error" not in response else "Error: Unable to classify difficulty."

# Function to evaluate the question, categorize it, and suggest a strategy
def evaluate_question(question):
    category = classify_question_category(question)
    is_rag_useful = should_use_rag_with_gemini(category)
    difficulty_response = classify_difficulty(question, category)
    is_illinois_law = "Illinois" in question

    print(f"Reason-4: Is the question related to Illinois law?\nAnswer-4: {'Yes' if is_illinois_law else 'No'}")

    if is_rag_useful:
        if is_illinois_law :
            high_level_response = should_use_raptor(question)
            strategy = "LLM + Raptor"
        else:
            strategy = "LLM + Google Search"
    else:
        if "Easy" in difficulty_response:
            strategy = "LLM"
        else:
            strategy = f"LLM + Google Search"


    print(f"Final Strategy: {strategy}")
    return strategy, category, is_rag_useful, difficulty_response


# Test the evaluation function with a sample question
if __name__ == "__main__":
    test_question = "What are the rules governing personal jurisdiction in Illinois?"
    final_strategy = evaluate_question(test_question)

    # Execute based on strategy
    if "LLM only" in final_strategy or "LLM + self-reflection" in final_strategy:
        response = generate_response(generative_model_gemini_15_pro, test_question)
        print(f"Response from Gemini 1.5 Pro: {response}")



