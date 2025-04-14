from flask import Flask, request, render_template
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from src.prompt import system_prompt
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# ✅ Load embeddings and FAISS index
print("Loading Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

print("Loading FAISS index...")
faiss_index = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # ⚠️ Only enable if you trust the index source!
)

# ✅ Create retriever from FAISS
retriever = faiss_index.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}  # Top 1 similar result
)

# ✅ Initialize Ollama LLM with your system prompt (make sure ollama serve is running)
llm = OllamaLLM(
    model="orca-mini",  # or "tinyllama"
    base_url=os.getenv("OLLAMA_BASE_URL"),
    system=system_prompt
)

# ✅ Create RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce"
)

# ==============================
# ✅ FLASK ROUTES
# ==============================

@app.route("/")
def index():
    """Render the chatbot frontend."""
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """Process user message and return bot response."""
    try:
        user_message = request.form["msg"]
        print(f"📝 User: {user_message}")

        # ✅ Use the correct input structure for invoke()
        response = rag_chain.invoke({"query": user_message})

        print(f"🤖 Full Response: {response}")

        # ✅ Return only the 'result' key content
        return response.get('result', "Sorry, I couldn't find an answer.")

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Something went wrong. Please try again."


# ==============================
# ✅ RUN FLASK APP
# ==============================

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)