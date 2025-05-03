
import os
import re
import time
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import txt2md
import pdf2md
import docx2md
import pptx2md
import img2md
import faiss_embedder
from pdf2md import summarize_markdown_groups
from anthropic import Anthropic
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
import structured_data
import sqlite3

# --- Configuration ---
dump_path = 'dump'
structured_db_path = os.path.join(dump_path, 'structured_db')
merged_summary_path = os.path.join(dump_path, 'merged_summaries.csv')
os.makedirs(dump_path, exist_ok=True)
os.makedirs(structured_db_path, exist_ok=True)

# Hardcoded API key
anthropic_api_key = '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'
anthropic = Anthropic(api_key=anthropic_api_key)
gemini_api = os.getenv('GEMINI_API_KEY', 'AIzaSyBS2npulOMMZ9WRj7b-UpoYHXVSa0Jju4o')
mistral_api_key = os.getenv('MISTRAL_API_KEY', 'VrAMhIHO61FjHTAYeibtLmla52bWnorV')

# Globals for structured data
structured_index = None
current_db_path = None
structured_dfs = []
structured_names = []
combined_structured_df = None
pd_agent = None
_indexed_files = {}

def minimal_ingest_documents(paths):
    """
    Ultra-fast document ingestion that skips entity extraction and Neo4j operations.
    Only creates FAISS indices for vector search.
    
    Args:
        paths: List of document paths to ingest
    """
    # No Neo4j operations in minimal mode
    print("Using minimal processing mode - skipping Neo4j operations")
    
    for idx, path in enumerate(paths):
        base = str(idx)
        md_path = os.path.join(dump_path, f"{base}.md")
        
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
            
        # Track the indexed file
        _indexed_files[path] = {
            'md_path': md_path,
            'index_time': time.time(),
            'original_path': path
        }

    for idx in range(len(paths)):
        md_file = os.path.join(dump_path, f"{idx}.md")
        try:
            with open(md_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            item = {'filename': os.path.basename(md_file), 'content': content, 'document_number': idx}
            chunks = faiss_embedder.chunk_and_index(item)
            
            # Use minimal processing for maximum speed
            faiss_embedder.index_chunks(
                chunks, 
                api_key=anthropic_api_key, 
                process_urls=True,
                process_entities=False,
                minimal_processing=True
            )
        except Exception as e:
            print(f"Error indexing {md_file}: {e}")

# --- Enhanced Data Query Handler ---
def handle_data_queries(df: pd.DataFrame, question: str):
    question_l = question.lower()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Superlative: min/max
    super_match = re.search(r"\b(highest|maximum|max|lowest|minimum|min)\b", question_l)
    if super_match:
        for col in num_cols:
            if col.lower() in question_l:
                if 'min' in question_l or 'lowest' in question_l:
                    val = df[col].min()
                    label = 'minimum'
                else:
                    val = df[col].max()
                    label = 'maximum'
                rows = df[df[col] == val]
                return f"{label.title()} of '{col}' is {val}.\nRows:\n{rows.to_string(index=False)}"

    # Aggregation: average/mean
    if 'average' in question_l or 'mean' in question_l:
        for col in num_cols:
            if col.lower() in question_l:
                avg = df[col].mean()
                return f"Average of '{col}' is {avg:.2f}."

    return None

# === Ingestion: Documents ===
def ingest_documents(paths):
    for idx, path in enumerate(paths):
        base = str(idx)
        md_folder = os.path.join(dump_path, base)
        md_path = os.path.join(dump_path, f"{base}.md")

        if path.endswith('.txt'):
            txt2md.generate_md(path, anthropic_api_key, md_folder)
        elif path.endswith('.pdf'):
            pdf2md.generate_md(path, anthropic_api_key, md_folder)
        elif path.endswith('.docx'):
            docx2md.generate_md(path, anthropic_api_key, md_folder)
        elif path.endswith('.pptx'):
            pptx2md.generate_md(path, anthropic_api_key, md_folder)
        else:
            img2md.generate_md(path, anthropic_api_key, md_folder)

        _indexed_files[path] = {
            'md_path': md_path,
            'index_time': time.time(),
            'original_path': path
        }

    for idx in range(len(paths)):
        md_file = os.path.join(dump_path, f"{idx}.md")
        try:
            with open(md_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            item = { 'filename': os.path.basename(md_file), 'content': content, 'document_number': idx }
            chunks = faiss_embedder.chunk_and_index(item)
            faiss_embedder.index_chunks(chunks, api_key=anthropic_api_key)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error indexing {md_file}: {e}")

# === Ingestion: Structured Data ===
def ingest_structured(file_paths):
    global structured_index, current_db_path, combined_structured_df, pd_agent

    structured_dfs.clear()
    structured_names.clear()

    for p in file_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
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
        combined_structured_df = pd.concat(structured_dfs, ignore_index=True)
        db_name = f"structured_{int(time.time())}.db"
        current_db_path = os.path.join(structured_db_path, db_name)

        structured_index = structured_data.index_structured(
            structured_dfs, structured_names, current_db_path
        )
        with open(os.path.join(structured_db_path, 'current_db.txt'), 'w') as f:
            f.write(current_db_path)

        anth_agent = ChatAnthropic(
            model='claude-3-sonnet-20240229',
            api_key=anthropic_api_key,
            temperature=0
        )
        pd_agent = create_pandas_dataframe_agent(
            anth_agent,
            combined_structured_df,
            verbose=False,
            allow_dangerous_code=True
        )

# === Remove from Memory ===
def remove_from_memory(filepath):
    if filepath not in _indexed_files:
        print(f"File {filepath} not found in indexed files.")
        return False
    try:
        md_path = _indexed_files[filepath]['md_path']
        if os.path.exists(md_path):
            os.remove(md_path)
        folder = os.path.join(dump_path, os.path.basename(md_path).split('.')[0])
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        del _indexed_files[filepath]
        print(f"Removed {filepath} from memory")
        return True
    except Exception as e:
        print(f"Error removing {filepath}: {e}")
        return False

# === Load Structured ===
def load_structured():
    global structured_index, current_db_path
    try:
        path = open(os.path.join(structured_db_path, 'current_db.txt')).read().strip()
        structured_index = structured_data.load_existing_index(path)
        current_db_path = path
    except Exception:
        print("No existing structured DB to load.")

# === Query Structured via SQL ===
def query_structured_data(sql):
    if not structured_index:
        print("No structured DB loaded.")
        return pd.DataFrame()
    return structured_index.query(sql)

# --- Pipeline Nodes ---
SCHEMA_DESC = (
    "Access to stores:\n"
    "[ENTITY] Semantic Index\n"
    "[DATA] Structured SQL or DataFrame operations\n"
    "[GENERAL] Section-level or hybrid search\n"
    "[DETAILED] Paragraph-level Index\n"
    "[VAGUE] Ambiguous"
)
CLASSIFY_INSTR = 'Classify: ENTITY, DATA, GENERAL, DETAILED, VAGUE.'
FEW_SHOT = [
    {"role": "user", "content": "Who is the CEO of Acme Corp?"},
    {"role": "assistant", "content": "ENTITY"},
    {"role": "user", "content": "What was Q1 2025 revenue in Europe?"},
    {"role": "assistant", "content": "DATA"},
]
ANSWER_PROMPT = (
    "Use context to answer:\n"
    "{context}\n"
    "Question: {question}"
)

def preprocess_query(state):
    state['processed_query'] = re.sub(r"[^\w\s]", "", state['question'].lower()).strip()
    return state

def describe_input_data(state):
    desc = []
    try:
        md_files = [f for f in os.listdir(dump_path) if f.endswith('.md')]
        for md in md_files[:3]:
            txt = open(os.path.join(dump_path, md), 'r', encoding='utf-8', errors='replace')
            desc.append(txt.read(100).replace('\n', ' '))
    except:
        pass
    struct_info = []
    if structured_dfs:
        for name, df in zip(structured_names, structured_dfs):
            info = f"{name}: {df.shape[0]} rows x {df.shape[1]} cols"
            sample = df.head(3).to_dict('records')
            struct_info.append(f"{info}, sample={sample}")
    state['input_description'] = ''
    if desc:
        state['input_description'] = ' | '.join(desc)
    if struct_info:
        state['input_description'] += '\nStructured Data: ' + ' | '.join(struct_info)
    return state

def classify_node(state):
    messages = [
        {"role":"user","content": SCHEMA_DESC + '\n' + CLASSIFY_INSTR + '\n' + ''.join(m['content'] + '\n' for m in FEW_SHOT) + state['question']}
    ]
    try:
        r = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            system="Classify user queries into route categories.",
            messages=messages,
            max_tokens=50
        )
        tag = r.content[0].text.strip().upper()
    except:
        tag = 'GENERAL'
    state['route'] = tag if tag in ['ENTITY','DATA','GENERAL','DETAILED','VAGUE'] else 'GENERAL'
    return state

def retrieval_node(state):
    state.pop('agent_answer', None)
    q = state['processed_query']
    route = state['route']

    if route == 'ENTITY':
        state['retrieved'] = faiss_embedder.section_faiss_search(q, api_key=anthropic_api_key)
    elif route == 'DATA':
        df = query_structured_data(q)
        if df.empty and combined_structured_df is not None:
            result = handle_data_queries(combined_structured_df, state['question'])
            if result:
                state['agent_answer'] = result
                return state
            mask = combined_structured_df.astype(str).apply(
                lambda col: col.str.contains(q, case=False, na=False)
            )
            df = combined_structured_df[mask.any(axis=1)]
        if df.empty and pd_agent:
            state['agent_answer'] = pd_agent.run(state['question'])
        else:
            state['retrieved'] = df
    else:
        state['retrieved'] = faiss_embedder.section_faiss_search(q, api_key=anthropic_api_key)

    return state

def answer_and_fallback_node(state):
    if 'agent_answer' in state:
        return {'answer': state.pop('agent_answer')}

    ctx = state.get('retrieved', '')
    if isinstance(ctx, pd.DataFrame):
        ctx = ctx.to_string(index=False)

    prompt = state.get('input_description', '') + '\n' + ANSWER_PROMPT.format(
        context=ctx, question=state['question']
    )
    try:
        r = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            system="Answer user questions using the provided context.",
            messages=[{"role":"user","content":prompt}],
            max_tokens=400
        )
        ans = r.content[0].text.strip()
    except:
        ans = ''

    if not ans or len(ans) < 20 or 'might' in ans.lower():
        state = classify_node(state)
        state = retrieval_node(state)
        return answer_and_fallback_node(state)

    return {'answer': ans}

# Build pipeline
builder = StateGraph(input='question', output='answer')
builder.add_node('preprocess', preprocess_query)
builder.add_node('describe', describe_input_data)
builder.add_node('classify', classify_node)
builder.add_node('retrieve', retrieval_node)
builder.add_node('answer', answer_and_fallback_node)
builder.set_entry_point('preprocess')
builder.add_edge('preprocess', 'describe')
builder.add_edge('describe', 'classify')
builder.add_edge('classify', 'retrieve')
builder.add_edge('retrieve', 'answer')
builder.add_edge('answer', END)

pipeline = builder.compile()

# Main loop
def main():
    while True:
        print("1. Ingest Docs | 2. Ingest Structured | 3. Load Structured | 4. Ask Questions | 5. Run SQL | 6. Exit")
        choice = input('> ').strip()
        if choice == '1':
            paths = []
            print("Enter document paths (blank to finish):")
            while True:
                p = input('Path: ').strip()
                if not p:
                    break
                paths.append(p)
            ingest_documents(paths)
        elif choice == '2':
            files = []
            print("Enter structured file paths (blank to finish):")
            while True:
                f = input('File: ').strip()
                if not f:
                    break
                files.append(f)
            ingest_structured(files)
        elif choice == '3':
            load_structured()
        elif choice == '4':
            print("Enter questions ('exit' to return):")
            while True:
                q = input('Q: ').strip()
                if q.lower() == 'exit':
                    break
                print(pipeline.invoke({'question': q}).get('answer', 'No answer.'))
        elif choice == '5':
            sql = input('SQL: ').strip()
            df = query_structured_data(sql)
            print(df.to_string(index=False))
        elif choice == '6':
            break
        else:
            print("Invalid choice. Enter 1-6.")

if __name__ == '__main__':
    main()

