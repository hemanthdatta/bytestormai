import os
import pandas as pd
import txt2md
import pdf2md
import docx2md
import pptx2md
import img2md
import faiss_embedder
from pdf2md import summarize_markdown_groups
from mistralai import Mistral
import csv
import time
import re
from tqdm import tqdm
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END
import sqlite3

# --- Configuration ---
dump_path = 'dump'
structured_db_path = os.path.join(dump_path, 'structured_db')
merged_summary_path = os.path.join(dump_path, 'merged_summaries.csv')
os.makedirs(dump_path, exist_ok=True)
os.makedirs(structured_db_path, exist_ok=True)

gemini_api = os.getenv('GEMINI_API_KEY', 'AIzaSyBS2npulOMMZ9WRj7b-UpoYHXVSa0Jju4o')
mistral_api_key = os.getenv('MISTRAL_API_KEY', 'VrAMhIHO61FjHTAYeibtLmla52bWnorV')

if not mistral_api_key.strip():
    raise ValueError("Valid Mistral API key required.")

mistral = Mistral(api_key=mistral_api_key)

# Globals for structured data
structured_index = None
current_db_path = None
structured_dfs = []                # list of DataFrames
structured_names = []              # corresponding table names
combined_structured_df = None
pd_agent = None

# === Ingestion: Documents ===
def ingest_documents(paths):
    for idx, path in enumerate(paths):
        base = str(idx)
        if path.endswith('.txt'):
            txt2md.generate_md(path, gemini_api, os.path.join(dump_path, base))
        elif path.endswith('.pdf'):
            pdf2md.generate_md(path, gemini_api, os.path.join(dump_path, base))
        elif path.endswith('.docx'):
            docx2md.generate_md(path, gemini_api, os.path.join(dump_path, base))
        elif path.endswith('.pptx'):
            pptx2md.generate_md(path, gemini_api, os.path.join(dump_path, base))
        else:
            img2md.generate_md(path, gemini_api, os.path.join(dump_path, base))

    for idx in range(len(paths)):
        md_file = os.path.join(dump_path, f"{idx}.md")
        try:
            with open(md_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            item = {'filename': os.path.basename(md_file), 'content': content, 'document_number': idx}
            chunks = faiss_embedder.chunk_and_index(item)
            faiss_embedder.index_chunks(chunks, api_key=mistral_api_key)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error indexing {md_file}: {e}")

# === Ingestion: Structured Data ===
def ingest_structured(file_paths):
    global structured_index, current_db_path
    global structured_dfs, structured_names, combined_structured_df, pd_agent
    structured_dfs.clear()
    structured_names.clear()

    for p in file_paths:
        try:
            name = os.path.splitext(os.path.basename(p))[0]
            # Support multiple file types
            if p.endswith('.csv'):
                df = pd.read_csv(p)
            elif p.endswith('.json'):
                df = pd.read_json(p)
            elif p.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(p)
            else:
                df = structured_data.parse_table(p)
            structured_dfs.append(df)
            structured_names.append(name)
        except Exception as e:
            print(f"Failed to parse {p}: {e}")

    if structured_dfs:
        # Combine into a single DataFrame for agent fallback
        combined_structured_df = pd.concat(structured_dfs, ignore_index=True)
        # Build/Overwrite structured DB
        db_name = f"structured_{int(time.time())}.db"
        db_path = os.path.join(structured_db_path, db_name)
        structured_index = structured_data.index_structured(structured_dfs, structured_names, db_path)
        current_db_path = db_path
        # Save current DB path
        with open(os.path.join(structured_db_path, 'current_db.txt'), 'w') as f:
            f.write(db_path)
        # Initialize a pandas-dataframe agent for fallback operations
        # Create a langchain ChatMistralAI model instance
        langchain_mistral = ChatMistralAI(
            model="mistral-small-latest",
            api_key=mistral_api_key,
            temperature=0
        )
        # Initialize a pandas-dataframe agent with the langchain_mistral model
        pd_agent = create_pandas_dataframe_agent(
            langchain_mistral, 
            combined_structured_df, 
            verbose=False,
            allow_dangerous_code=True
        )

# === Load Existing Structured Index ===
def load_structured():
    global structured_index, current_db_path
    try:
        path = open(os.path.join(structured_db_path, 'current_db.txt')).read().strip()
        structured_index = structured_data.load_existing_index(path)
        current_db_path = path
    except Exception:
        print("No existing structured DB to load.")

# === Query Structured Data by SQL ===
def query_structured_data(sql):
    if not structured_index:
        print("No structured DB loaded.")
        return pd.DataFrame()
    return structured_index.query(sql)

# === RAG Components ===
SCHEMA_DESC = '''Access to stores:
[ENTITY] Semantic Index
[DATA] Structured SQL or DataFrame operations
[GENERAL] Section-level or hybrid search
[DETAILED] Paragraph-level Index
[VAGUE] Ambiguous'''
CLASSIFY_INSTR = 'Classify: ENTITY, DATA, GENERAL, DETAILED, VAGUE.'
FEW_SHOT = [
    {"role":"user","content":"Who is the CEO of Acme Corp?"},
    {"role":"assistant","content":"ENTITY"},
    {"role":"user","content":"What was Q1 2025 revenue in Europe?"},
    {"role":"assistant","content":"DATA"},
]

def preprocess_query(state):
    state['processed_query'] = re.sub(r"[^\w\s]", "", state['question'].lower()).strip()
    return state

# Include both text chunks and structured schema description
def describe_input_data(state):
    desc = []
    try:
        md_files = [f for f in os.listdir(dump_path) if f.endswith('.md')]
        for md in md_files[:3]:
            with open(os.path.join(dump_path, md), 'r', encoding='utf-8', errors='replace') as f:
                desc.append(f.read()[:100].replace('\n',' '))
    except Exception as e:
        print(f"Error reading markdown: {e}")

    struct_info = []
    if structured_dfs:
        for name, df in zip(structured_names, structured_dfs):
            struct_info.append(f"{name}: columns {list(df.columns)} ({len(df)} rows)")

    state['input_description'] = ''
    if desc:
        state['input_description'] += ' | '.join(desc)
    if struct_info:
        state['input_description'] += '\nStructured Data: ' + ' | '.join(struct_info)
    return state

def classify_node(state):
    msgs = [{"role":"system","content":SCHEMA_DESC},
            {"role":"system","content":CLASSIFY_INSTR}] + FEW_SHOT + [{"role":"user","content":state['question']}]
    r = mistral.chat.complete(model="mistral-small-latest", messages=msgs, temperature=0)
    tag = r.choices[0].message.content.strip().upper()
    state['route'] = tag if tag in ['ENTITY','DATA','GENERAL','DETAILED','VAGUE'] else 'GENERAL'
    return state

# Hybrid retrieval for both text and structured
def retrieval_node(state):
    state.pop('agent_answer', None)
    q = state['processed_query']
    route = state['route']
    if route == 'ENTITY':
        state['retrieved'] = faiss_embedder.section_faiss_search(q, api_key=mistral_api_key)
    elif route == 'DATA':
        df = query_structured_data(q)
        if df.empty and pd_agent:
            state['agent_answer'] = pd_agent.run(state['question'])
        else:
            state['retrieved'] = df
    else:
        state['retrieved'] = faiss_embedder.section_faiss_search(q, api_key=mistral_api_key)
    return state

def format_context(ret):
    if isinstance(ret, pd.DataFrame):
        return ret.to_string(index=False)
    return str(ret)

ANSWER_PROMPT = '''Use context to answer:
{context}
Question: {question}'''

def answer_and_fallback_node(state):
    if state.get('agent_answer'):
        ans = state.pop('agent_answer')
        return {'answer': ans}
    ctx = format_context(state.get('retrieved', ''))
    full_prompt = f"{state.get('input_description','')}\n" + ANSWER_PROMPT.format(context=ctx, question=state['question'])
    for i in range(5):
        try:
            r = mistral.chat.complete(model="mistral-large-latest", messages=[{"role":"user","content":full_prompt}])
            ans = r.choices[0].message.content.strip()
            break
        except Exception:
            time.sleep(2**i)
    else:
        ans = ""
    if not ans or len(ans) < 20 or 'might' in ans.lower():
        state = classify_node(state)
        state = retrieval_node(state)
        return answer_and_fallback_node(state)
    return {'answer': ans}

# === Pipeline Build ===
builder = StateGraph(input='question', output='answer')
builder.add_node('preprocess', preprocess_query)
builder.add_node('describe', describe_input_data)
builder.add_node('classify', classify_node)
builder.add_node('retrieve', retrieval_node)
builder.add_node('answer', answer_and_fallback_node)
builder.set_entry_point('preprocess')
builder.add_edge('preprocess','describe')
builder.add_edge('describe','classify')
builder.add_edge('classify','retrieve')
builder.add_edge('retrieve','answer')
builder.add_edge('answer', END)

pipeline = builder.compile()

# === CLI ===
def main():
    while True:
        print("1. Ingest Docs | 2. Ingest Structured | 3. Load Structured | 4. Ask Questions | 5. Run SQL | 6. Exit")
        choice = input('> ').strip()
        if choice == '1':
            paths = []
            print("Enter document paths (blank line to finish):")
            while True:
                p = input('Path: ').strip()
                if not p:
                    break
                paths.append(p)
            ingest_documents(paths)
        elif choice == '2':
            files = []
            print("Enter structured file paths (blank line to finish):")
            while True:
                f = input('File: ').strip()
                if not f:
                    break
                files.append(f)
            ingest_structured(files)
        elif choice == '3':
            load_structured()
        elif choice == '4':
            print("Enter your questions (type 'exit' to return):")
            while True:
                q = input('Q: ').strip()
                if q.lower() == 'exit':
                    break
                result = pipeline.invoke({'question': q})
                print(result.get('answer', 'No answer.'))
        elif choice == '5':
            sql = input('SQL: ').strip()
            df = query_structured_data(sql)
            print(df.to_string(index=False))
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please select a number from 1 to 6.")

if __name__ == '__main__':
    main()
