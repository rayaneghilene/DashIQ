# DashIQ
*"Turn Chaos into clarity with actionnable insights strainght from your data"*

This repository contains the code for a conversational AI tool designed to assist users in uploading unstructured and noisy CSV/Excel files and receiving clear, actionable insights. Through a user-friendly chat interface, the system leverages large language models (LLMs) and intelligent data processing techniques to make sense of raw data, cleaning it, summarizing key information, identifying issues, and providing recommendations.


## Abstract & Research Axes

This system opens up multiple avenues for applied research in data preprocessing, interaction design, and language model optimization. Below are several potential research directions:

1. **Automated Cleaning of Semi-Structured Spreadsheets**  
   Investigate hybrid rule-based and generative models for noisy file parsing, column name normalization, and unit/datetime standardization. 

   We will start by building upon the work done in the paper ["TableLlama: Towards Open Large Generalist Models for Tables"](https://arxiv.org/pdf/2311.09206)

2. **Missing Data Handling via Conversational Feedback**  
   Explore context-aware imputation methods augmented by user clarification, preference learning, or model introspection.

3. **Explainable AI for Noisy Tabular Data**  
   Enable traceable, user-guided decisions about data transformations and insight generation, incorporating SHAP, LIME, or contrastive examples.

4. **Context Packing for LLM Training on Tabular Data**  
   Study novel context-packing strategies that combine structured metadata with natural language descriptions for improved finetuning or few-shot prompting.
   
   Paper inspiration: [Structured Packing in LLM Training Improves Long Context Utilization](https://arxiv.org/abs/2312.17296)

5. **Interactive Data Visualization via Language Models**  
   Assess prompt-based chart generation tools (e.g., Altair, Plotly) integrated into a chat UX for dynamic data storytelling.

6. **Preference Optimization and RLHF in Chat Interfaces**  
   Use Reinforcement Learning from Human Feedback (RLHF) or preference ranking to fine-tune the assistant's ability to generate relevant insights and reduce hallucinations.

   Ressources : 
   - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
   - [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

7. **Multimodal Fusion of Text and Tables**  
   Integrate vision-language models / OCR to parse scanned spreadsheets or annotated PDF tables, extending functionality beyond CSV/EXCEL.


## Usage
### Installation

```bash
git clone https://github.com/yourusername/csv-chat-insights.git
cd csv-chat-insights
```

### Docker 🐳

```sh 
docker build -t dashiq .
docker run -p 8501:8501 dashiq
```

### Python

- **Install requirements:**

```bash
pip install -r requirements.txt
```

- **Run Steramlit app:**
```bash
streamlit run app.py
```

## Preview of the App:
The user can upload CSV and PDF documents, The files are first cleaned and preprocessed, stored in a vectore store and passed into the model as context. The user can then  

> [!NOTE]  
> This is a dummy version, we use Mistral ai to run the analysis, a better version with cutom foundation models is coming soon :) 

![image](https://github.com/user-attachments/assets/8051f9bc-c33d-4225-b60e-307015a2917e)
![image](https://github.com/user-attachments/assets/e6fb1ba4-d7dd-4d0d-8c4b-89b58acd37f2)
![image](https://github.com/user-attachments/assets/271055d4-1441-4ef8-8628-8b82ec6ac23d)



## Contributing
Feel free to open issues or submit pull requests for bug fixes, new features, or research collaborations.

## Contact
Feel free to reach out about any questions/suggestions at rayane.ghilene@ensea.fr