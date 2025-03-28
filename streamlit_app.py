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
import uuid
from datetime import datetime
from pathlib import Path
from firebase_config import initialize_firebase, get_all_firestore_records, upload_to_firestore

PROMPT_FILE='data/prompts.csv'
RESULT_FILE='data/results.csv'
DB_RESULT_FILE = 'results_db.csv'

current_date = datetime.now().strftime("%Y-%m-%d")

def upload_results(model_name, prompt, resp, time_elapsed, run_date):
    record =  {
        'id' : str(uuid.uuid4()),
        'model': model_name,
        'prompt': prompt,
        'response': resp,
        'time_elapsed': time_elapsed,
        'run_date': run_date,
    }
    upload_to_firestore(record)

def initialize_models():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    os.environ["GOOGLE_API_KEY"]=st.secrets["GEMINI_API_KEY"]
    os.environ["XAI_API_KEY"]=st.secrets["XAI_API_KEY"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"]=st.secrets["HF_TOKEN"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    models={
        "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
        "claude-sonnet": ChatAnthropic(model="claude-3-5-sonnet-20240620"),
        "gemini-1.5-pro": ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
        "grok-2-latest": ChatXAI(model="grok-2-latest"),
        #"llama": HuggingFaceEndpoint(repo_id="meta-llama/Llama-2-70b-chat-hf",task="text-generation",max_new_tokens=512,temperature=0.7),
        "llama":ChatGroq(model="llama-3.3-70b-versatile"),
        "deepseek":ChatGroq(model="deepseek-r1-distill-llama-70b"),
        #"gpt-4o": ChatOpenAI(model="gpt-4o"),
        #"claude-haiku": ChatAnthropic(model="claude-3-haiku-20240307"),
    }
    return models

def apply_model(model,user_input):
    try:
        start_time=time.time()
        user_message=HumanMessage(content=f"{user_input}")
        messages = [user_message]
        response=model.invoke(messages)
        end_time=time.time()
        elapsed_time=end_time-start_time
        return response.content, elapsed_time
    except Exception as e:
        return f"Error: {e}",0
    
def read_file_from_ui_or_fs():
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a prompt file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv(PROMPT_FILE,index=False)
        return df
    else:
        if os.path.exists(PROMPT_FILE):
            df=pd.read_csv(PROMPT_FILE)
            return df
    return None

def show_download_sidebar():
    with st.sidebar:
        file_path = Path(RESULT_FILE)
        if file_path.exists():
            st.divider()
            local_file_name = st.text_input("File Name for Download", value = "complete_set.csv")
            with open(RESULT_FILE, "rb") as file:
                file_bytes = file.read()
            st.download_button(label="Download",data=file_bytes,file_name= local_file_name,mime="text/csv")
            if st.button("Clear file"):
                os.remove(RESULT_FILE)
        
        records = get_all_firestore_records()
        if records:
            df = pd.DataFrame(records)
            df.drop(columns = ['id', 'doc_id'], inplace = True)
            st.divider()
            db_file_name = st.text_input("File Name for Download", value = DB_RESULT_FILE)
            st.download_button(
                label = "Download from database",
                data = df.to_csv(index = False),
                file_name = db_file_name,
                mime = "text/csv"
            )


def save_results(count,result_df):
    if count>0:
        if os.path.exists(RESULT_FILE):
            result_df.to_csv(RESULT_FILE,mode='a', index=False, header=False)
        else:
            result_df.to_csv(RESULT_FILE, index=False)
def run_all_models(df,model_list,run_count):
    update_ui.write(f"Starting run with {run_count=} for {models=}")
    current_count=0
    total_count=run_count*len(model_list)*df.shape[0]
    progress_bar=st.progress(current_count)
    results=[]
    with st.spinner("Running...", show_time=True):
        for i in range(1,run_count+1):
            for model_choice in model_list.keys():
                for _,row in df.iterrows():
                    user_input=row['Prompt']
                    update_ui.write(f"Completed {current_count}/{total_count} steps. {i=}, {model_choice=}, {user_input=}")
                    rsp,elapsed_time = apply_model(models[model_choice],user_input)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    results.append({'model':model_choice,'prompt':user_input,'response':rsp,'time_seconds':elapsed_time,'current_date':current_date})
                    upload_results(model_choice, user_input, rsp, elapsed_time, current_date)
                    current_count+=1
                    progress_bar.progress( (1.0*current_count) / total_count)
    update_ui.write(f"All done!!")
    result_df=pd.DataFrame(results)
    st.dataframe(result_df,hide_index=True)
    save_results(current_count,result_df)
#
# Main
#
st.title("Sentio")
initialize_firebase()
records = get_all_firestore_records()  
if records:
    df = pd.DataFrame(records)
st.dataframe(df)
update_ui=st.empty()
df = read_file_from_ui_or_fs()
models=initialize_models()
if df is not None:
    with st.sidebar.expander("Prompts"):
        st.dataframe(df, hide_index=True)
    options = st.sidebar.multiselect('Models',options=list(models.keys()),default=list(models.keys()))
    model_list={key:models[key] for key in options}
    run_count=st.sidebar.number_input("Runs",min_value=1,max_value=100, value=1)
    if st.sidebar.button("Run"):
        run_all_models(df,model_list,run_count)
show_download_sidebar()


















