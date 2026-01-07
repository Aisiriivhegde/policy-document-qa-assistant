import os
from PyPDF2 import PdfReader
from groq import Groq
from flask import Flask, request, jsonify

# Load Groq API Key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Flask App
app = Flask(__name__)
# Extract text from PDF
def extract_policy_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i in range(min(2, len(reader.pages))):  # Only first 2 pages
        text += f"\n\n[Policy Section - Page {i+1}]\n"
        page_text = reader.pages[i].extract_text() or ""
        text += page_text
    return text

POLICY_TEXT = extract_policy_text("policy_docs/sample_policy.pdf")

# Ask Groq (Llama 3.1)
def ask_model(question, policy_text):
    system_prompt = """
You are an insurance policy assistant.
Explain policy terms in very simple language.
Always reference the specific policy section or page number.
If the information is not present, say clearly that it is not mentioned.
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
POLICY DOCUMENT:
{policy_text}

QUESTION:
{question}
"""
            }
        ]
    )

    return completion.choices[0].message.content

# API Endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    answer = ask_model(question, POLICY_TEXT)
    return jsonify({"answer": answer})

@app.route("/", methods=["GET"])
def home():
    return """
    <h1>Policy QA Assistant</h1>
    <p>API working! Use POST /ask with JSON {"question": "..."}</p>
    <p>Test with PowerShell: Invoke-RestMethod -Uri "http://127.0.0.1:5000/ask" -Method Post -ContentType "application/json" -Body '{"question":"waiting period"}'</p>
    """

# Run App
if __name__ == "__main__":
    app.run(debug=True)

