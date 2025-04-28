import streamlit as st
import requests

# Base URL for the API backend
API_BASE = "http://localhost:5000"

# Title of the Streamlit app
st.title("📊 CSV Knowledge Chatbot")

# File uploader for uploading CSV files
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
if uploaded_file:
    # Send the uploaded file to the API backend
    files = {'file': uploaded_file}
    response = requests.post(f"{API_BASE}/upload", files=files)
    if response.status_code == 200:
        st.success("✅ File uploaded and indexed successfully!")
    else:
        st.error("❌ Failed to upload file.")

# Fetch the list of available CSV files from the backend
csvs = requests.get(f"{API_BASE}/list_csvs").json()

# Dropdown to select a CSV file for querying
selected_csv = st.selectbox("📂 Select a CSV file to chat with:", csvs)

# Text input for the user to ask questions about the selected CSV
query = st.text_input("💬 Ask a question about the selected CSV:")
if st.button("Ask") and query and selected_csv:
    # Send the query and selected CSV filename to the API backend
    response = requests.post(f"{API_BASE}/chat", json={
        "question": query,
        "filename": selected_csv
    })
    if response.status_code == 200:
        # Display the answer returned by the backend
        st.markdown(f"🧠 **Answer:** {response.json()['answer']}")
    else:
        st.error("⚠️ Error getting answer.")
