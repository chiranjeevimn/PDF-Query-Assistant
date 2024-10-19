from flask import Flask, render_template, request, jsonify, session
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
import PyPDF2

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management


# Initialize the LLM
def get_api_key():
    """Get the API key from Gemini"""
    api = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    return api


def initialize_llm(api_key):
    """Initialize the GoogleGenerativeAI LLM with safety settings."""
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )


# Extract PDF content
def extract_pdf_content(pdf_file):
    """Extract text from the uploaded PDF file."""
    reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ''
    for page in reader.pages:
        pdf_text += page.extract_text()
    return pdf_text


# Check relevance of the question
def is_question_relevant(question, pdf_content):
    """Check if the question is relevant to the PDF content."""
    keywords = pdf_content.split()
    question_words = question.split()
    return any(word in keywords for word in question_words)


# Get response from LLM
def get_gemini_response(pdf_content, question):
    # Initialize the custom Gemini LLM
    API_KEY = get_api_key()
    gemini_llm = initialize_llm(API_KEY)

    # Define a prompt template that includes the PDF content and question
    prompt_template = PromptTemplate(
        input_variables=["question", "pdf_content"],
        template="Answer the question based on the following document:\n\n{pdf_content}\n\nQuestion:\n{question}"
    )

    # Check if the question is relevant to the PDF content
    if is_question_relevant(question, pdf_content):
        # Format the prompt with the question and PDF content
        formatted_prompt = prompt_template.format(question=question, pdf_content=pdf_content)

        try:
            # Invoke the LLM to get the answer
            response = gemini_llm.invoke(formatted_prompt)
        except Exception as e:
            print(f"Error getting response from LLM: {e}")
            return "I'm sorry, there was an error processing your request."

        return response
    else:
        return "I can only answer questions related to the PDF."


@app.route("/", methods=["GET", "POST"])
def home():
    """Home route to handle chat input and display history."""
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        if 'pdf_file' in request.files:
            # Handle PDF file upload
            pdf_file = request.files['pdf_file']
            session['pdf_content'] = extract_pdf_content(pdf_file)
            return jsonify({"success": True, "message": "PDF uploaded successfully!"})

        # Handle user question
        user_question = request.form["user_question"]
        pdf_content = session.get('pdf_content', "")

        if not pdf_content:
            return jsonify({"success": False, "response": "Please upload a PDF file first."})

        response = get_gemini_response(pdf_content, user_question)

        # Add to chat history
        session['chat_history'].append({"user": user_question, "bot": response})
        return jsonify({"success": True, "response": response})

    return render_template("index.html", chat_history=session.get('chat_history', []))


@app.route("/reset", methods=["POST"])
def reset():
    """Reset chat history and PDF content."""
    session.pop('chat_history', None)
    session.pop('pdf_content', None)
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)
