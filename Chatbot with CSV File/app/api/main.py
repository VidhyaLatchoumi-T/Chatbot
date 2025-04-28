from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import glob
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Define folders for storing data and vector databases
DATA_FOLDER = "data"
VECTOR_DB_FOLDER = "vectorstore"

# Ensure the folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Endpoint to upload and process a CSV file
@app.route("/upload", methods=["POST"])
def upload_csv():
    # Check if a file is included in the request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file to the data folder
    file = request.files["file"]
    file_path = os.path.join(DATA_FOLDER, file.filename)
    file.save(file_path)

    # Load the CSV file using LangChain's CSVLoader
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Split the text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    # Generate embeddings for the text chunks
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Create a FAISS vector store and save it locally
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(os.path.join(VECTOR_DB_FOLDER, file.filename.split(".")[0]))

    return jsonify({"message": "File uploaded and indexed successfully"}), 200

# Endpoint to list all available CSV files
@app.route("/list_csvs", methods=["GET"])
def list_csvs():
    # Get all CSV filenames from the data folder
    csvs = [os.path.basename(path) for path in glob.glob(os.path.join(DATA_FOLDER, "*.csv"))]
    return jsonify(csvs)

# Endpoint to handle chat queries
@app.route("/chat", methods=["POST"])
def chat():
    # Extract the question and filename from the request
    data = request.json
    question = data.get("question")
    filename = data.get("filename")

    # Validate the inputs
    if not question or not filename:
        return jsonify({"error": "Filename and question are required"}), 400

    # Load the corresponding FAISS vector store
    db_path = os.path.join(VECTOR_DB_FOLDER, filename.split(".")[0])
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    # Initialize the language model
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.1
    )

    # Create a QA chain and get the answer
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    result = qa_chain({"question": question, "chat_history": []})

    return jsonify({"answer": result["answer"]})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
