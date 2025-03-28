import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from pydantic.v1 import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

PROMPT_FILE='data/prompts.csv'
RESULT_FILE='data/results.csv'

def initialize_models():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    os.environ["GOOGLE_API_KEY"]=st.secrets["GEMINI_API_KEY"]
    os.environ["XAI_API_KEY"]=st.secrets["XAI_API_KEY"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"]=st.secrets["HF_TOKEN"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    models = {
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
        "claude-sonnet": ChatAnthropic(model="claude-3-5-sonnet-20240620"),
        #"gemini-1.5-pro": ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
        #"grok-2-latest": ChatXAI(model="grok-2-latest"),
        #"llama": HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-70b-chat-hf",task="text-generation",max_new_tokens=512,temperature=0.7),
        #"llama":ChatGroq(model="llama-3.3-70b-versatile"),
        #"deepseek":ChatGroq(model="deepseek-r1-distill-llama-70b")
        }
    return models
def show_download_sidebar():
    file_path = Path(RESULT_FILE)
    if file_path.exists:
        with st.sidebar:
            st.divider()
            with open(RESULT_FILE, "rb") as file:
                file_bytes = file.read()
            st.download_button(label = "Download Results", data = file_bytes, file_name = "Responses.csv", mime = "text/csv")
            if st.button("Clear File"):
                os.remove(RESULT_FILE)
#tries to invoke message using given AI model and input
def apply_model(model, input):
    try:
        start_time = time.time()
        user_inp = HumanMessage(content = f"{input}")
        messages = [user_inp]
        resp = model.invoke(messages)
        end_time = time.time()
        return resp, end_time - start_time
    except Exception as e:
        return f"Error: {e}", 0
    
def read_file_from_ui_or_fs():
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload file here", type = ['csv'])
    if uploaded_file != None:
        data = pd.read_csv(uploaded_file)
        data.to_csv(PROMPT_FILE, index = False)
        return data
    if os.path.exists(PROMPT_FILE):
        data = pd.read_csv(PROMPT_FILE)
        return data
    return None

def save_all_responses(responses):
    if len(responses) < 1:
        return
    df = pd.DataFrame(responses)
    if os.path.exists(RESULT_FILE):
        df.to_csv(RESULT_FILE, mode = 'a', index = False)
    else:
        df.to_csv(RESULT_FILE, index = False)

def run_all_models(df, model_list, run_count, models):
    st.write(f"{df=}, {model_list=}, {run_count=}")
    responses = []
    for i in range(1, run_count + 1):
        for model in model_list:
            for _, row in df.iterrows():
                user_input = row['Prompt']
                response, time_taken = apply_model(models[model], user_input)
                responses.append({"Prompt": user_input, "Model": model, "Response": response.content, "Time": time_taken})
                st.write(f"{user_input=}, {model=}, {run_count=}, {response.content=}, {time_taken=}")
    save_all_responses(responses)

def main():
    models = initialize_models()
    st.write("Models initialized:", list(models.keys()))
    st.title("Sentio")
    update_ui=st.empty()
    df = read_file_from_ui_or_fs()
    if df is not None:
        st.sidebar.dataframe(df, hide_index = True)
        options = st.sidebar.multiselect("Models", options = models.keys(), default = list(models.keys()))
        model_list = {key: models[key] for key in options}
        run_count = st.sidebar.number_input("Runs", min_value = 1, max_value = 100)
        if st.sidebar.button("Run"):
            run_all_models(df, model_list, run_count, models)
        show_download_sidebar()




if __name__ == "__main__":
    main()
