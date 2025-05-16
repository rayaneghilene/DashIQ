from io import StringIO
import csv
import re
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_community.llms import Ollama  
from langchain.chains import RetrievalQA
import tempfile
import os
import matplotlib.pyplot as plt
import plotly.express as px
from mistralai import Mistral

mistral_api_key = os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_API_KEY")

model = "mistral-large-latest"

client = Mistral(api_key=mistral_api_key)


def process_csv(path):
    """
    Loads and splits a CSV file from a given file path using LangChain's CSVLoader.
    """
    loader = CSVLoader(path)
    docs = loader.load_and_split()
    return docs

def format_prompt(instruction: str, input_text: str, question: str) -> str:
    """
    Format the prompt using the given instruction, input, and question.

    Parameters:
    - instruction (str): The task instruction.
    - input_text (str): Context or input for the task.
    - question (str): The question to be answered.

    Returns:
    - str: A formatted prompt string.
    """
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{input_text}\n\n"
        # "### Question:\n"
        # f"{question}\n\n"
        "### Response:"
    )
    return prompt

def parse_llm_csv_output(llm_output: str) -> pd.DataFrame:
    """
    Parses CSV-like LLM output into a pandas DataFrame, assuming semicolon separators.
    """
    # Strip triple backticks if present
    match = re.search(r"```csv\s*(.*?)\s*```", llm_output, re.DOTALL)
    if not match:
        print("No CSV block found.")
        return pd.DataFrame()
    
    csv_content = match.group(1)

# Attempt parsing with common delimiters
    for sep in [';', ',']:
        try:
            df = pd.read_csv(
                StringIO(csv_content),
                sep=sep,
                engine="python",
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines='skip'  # Skips lines that break parsing
            )
            if df.shape[1] > 1:  # Heuristic: ensure it's actually tabular
                return df
        except Exception as e:
            continue  # Try next delimiter

    print("Failed to parse CSV with known delimiters.")
    return pd.DataFrame()





def clean_csv(df_path):
    
    docs = process_csv(df_path)

    prompt = format_prompt(
        instruction="Organize the following csv data, Rename the columns if necessary, use ; as delimiter and use a ```csv``` to identify the table",
        input_text=str(docs),
        question=""
    )
    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    # print(chat_response_4.choices[0].message.content)
    llm_output =  chat_response.choices[0].message.content
    df_llm_output = parse_llm_csv_output(llm_output)
    print(df_llm_output.head())
    return df_llm_output