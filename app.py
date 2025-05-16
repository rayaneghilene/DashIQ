import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_community.llms import Ollama  # Uncomment for Ollama support
from langchain.chains import RetrievalQA
import tempfile
import os
import matplotlib.pyplot as plt
import plotly.express as px
from mistralai import Mistral

from Csv_Helper_func import clean_csv, process_csv, format_prompt, parse_llm_csv_output

# --------- SETUP SECTION ---------
# Set your Mistral API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_API_KEY")
model = "mistral-large-latest"

# --------- EMBEDDINGS SETUP ---------
@st.cache_resource
def get_embeddings():
    # Use a lightweight, open-source embedding model
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------- FILE PROCESSING ---------
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(file.read())
        loader = PyPDFLoader(tmp_pdf.name)
        docs = loader.load_and_split()
    os.unlink(tmp_pdf.name)
    return docs

# def process_csv(file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
#         tmp_csv.write(file.read())
#         clean_file = clean_csv(tmp_csv.name)
#         # loader = CSVLoader(tmp_csv.name)
#         loader = CSVLoader(clean_file)
#         docs = loader.load_and_split()
#     os.unlink(tmp_csv.name)
#     return docs
def process_csv(file_content: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', encoding='utf-8') as tmp_csv:
        tmp_csv.write(file_content)
        tmp_csv_path = tmp_csv.name

    clean_file = clean_csv(tmp_csv_path)
    loader = CSVLoader(clean_file)
    docs = loader.load_and_split()

    os.unlink(tmp_csv_path)
    return docs

# --------- VECTOR STORE (EMBEDDINGS INDEX) ---------
def create_vector_store(docs, embeddings):
    # Only embeddings are stored, not raw text
    return FAISS.from_documents(docs, embeddings)

# --------- LLM SETUP ---------
def get_llm():
    # --- MistralAI setup ---
    return ChatMistralAI(mistral_api_key=MISTRAL_API_KEY)
    # --- Ollama setup (uncomment to use) ---
    # return Ollama(model="mistral")  # Requires Ollama running locally

# --------- RAG PIPELINE ---------
def get_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

from langchain.schema import Document

# def docs_from_df(df: pd.DataFrame, source_name: str) -> list[Document]:
#     # flatten DataFrame to CSV text
#     csv_text = df.to_csv(sep=';', index=False)
#     # make it a single Document, or split by rows if you want finer granularity
#     return [Document(page_content=csv_text, metadata={"source": source_name})]

def docs_from_df(df):
    docs = []
    for i, row in df.iterrows():
        content = "\n".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(page_content=content, metadata={"row": i}))
    return docs


# --------- STREAMLIT DASHBOARD ---------
st.set_page_config(page_title="AI-Powered CSV & PDF Analyzer", layout="wide")
st.title("DashIQ")
st.write("Upload your CSV or PDF files. Ask questions and get clear, AI-generated explanations based on your data.")

uploaded_files = st.file_uploader("Upload CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True)
query = st.text_input("Ask a question about your data:")

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            st.info(f"Processing PDF: {file.name}")
            docs = process_pdf(file)
        elif file.name.endswith(".csv"):
            st.info(f"Processing CSV: {file.name}")
            # os.unlink(tmp_csv_path)
            # docs = clean_csv(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                tmp_csv.write(file.read())
                tmp_csv_path = tmp_csv.name
            
            cleaned_file = clean_csv(tmp_csv_path)
            initial_docs = docs_from_df(cleaned_file)
            docs = docs_from_df(cleaned_file)
            # loader = CSVLoader(initial_docs)
            # docs = loader.load_and_split()

            os.unlink(tmp_csv_path)
            # Process the CSV using the path
            # docs = process_csv(tmp_csv_path)
            # docs = process_csv(file)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue
        all_docs.extend(docs)

    if all_docs:
        embeddings = get_embeddings()
        vector_store = create_vector_store(all_docs, embeddings)
        llm = get_llm()
        rag_chain = get_rag_chain(vector_store, llm)

        if query:
            with st.spinner("Analyzing and generating insights..."):
                result = rag_chain({"query": query})
                st.subheader("AI Explanation")
                st.write(result["result"])
                with st.expander("Show source context"):
                    for doc in result["source_documents"]:
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                        st.write(doc.page_content)

        # --- Plotting Section ---
        csv_files = [file for file in uploaded_files if file.name.endswith(".csv")]
        if csv_files:
            st.subheader("📈 Data Plotting")
            selected_csv = st.selectbox("Select a CSV file to plot:", [file.name for file in csv_files])
            selected_file = next(file for file in csv_files if file.name == selected_csv)
            selected_file.seek(0)  # Reset file pointer
            df = pd.read_csv(selected_file)
            # df = clean_csv(selected_file)

            st.write("Preview of data:", df.head())

            plot_type = st.selectbox("Select plot type:", ["Line", "Bar", "Scatter"])
            columns = df.columns.tolist()
            x_col = st.selectbox("X-axis:", columns)
            y_col = st.selectbox("Y-axis:", columns)

            if st.button("Generate Plot"):
                fig = None
                if plot_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, title=f"{plot_type} plot of {y_col} vs {x_col}")
                elif plot_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{plot_type} plot of {y_col} vs {x_col}")
                elif plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{plot_type} plot of {y_col} vs {x_col}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid documents found in uploads.")

else:
    st.info("Please upload at least one CSV or PDF file.")
