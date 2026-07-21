#import
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.messages import ToolMessage

#define functions
@tool #decorative
def read_file(path: str):
    """use this tool to read PDF documents and extract their full text into local storage,
        for later chunking. do not return the full text into chat"""    
    clean = ""
    file = PdfReader(path)
    for idx, page in enumerate(file.pages):
        text = page.extract_text()
        if text.strip():
            clean += text + "\n"
    return f"Successfully read the file contents:\n{len(clean)}"

@tool
def make_db_chunks(text:str):
    """use this tool to break long documents into piece of chunks and then embed them to chromadb, only use it if u need to retrive a specific answer from a long text or to make a database, the text is picked up automatically"""
    spl = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 300)
    chunks = spl.split_text(text)
    Data_path = r"DataPath"
    chroma_path = r"ChromaPath"
    clientDB = chromadb.PersistentClient(path=Data_path)
    embedder = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    embedding = clientDB.get_or_create_collection(name="ChromaDataBase", embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    doc = []
    meta =[]
    id = []
    i= 0
    for chunk in chunks:
        doc.append(chunk)
        meta.append({
            "source": "deeplearning.pdf",
            "index": i
        })
        id.append(f"ID_{i}")
        i+=1
    upload_DB = embedding.upsert(
        documents=doc,
        metadatas=meta,
        ids=id
    )
    return f"successfully made {len(doc)} chunks"

@tool
def search(query):
    """use this tool when you have a specific question that needs to be extracted from chromadb created by make_db_chunks. do not modify the database, only search and retrieve"""
    Data_path = r"DataPath"
    clientDB = chromadb.PersistentClient(path=Data_path)
    embedder = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    embedding = clientDB.get_or_create_collection(name="ChromaDataBase", embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    answer = embedding.query(
        query_texts=[query],
        n_results=3
    )
    docs = answer["documents"][0]
    metas = answer["metadatas"][0]
    distances = answer["distances"][0]
    results = []
    for doc, meta, dist in zip(docs, metas, distances):
        results.append(
            f"chunk (source: {meta.get('source')}, index: {meta.get('index')}, distance: {dist:.3f}): {doc}"
        )

    return "\n\n".join(results)

@tool
def calculator(expression: str):
    """use this tool ONLY for mathematical equations expressions, do not use it for logic puzzles and common knowledge"""
    try:
        return str(eval(expression))
    except Exception as e:
        print("inavlid expression!")


#we do the steps outside of those functions like taking an inquiry and breaking it down b4 getting the final answer
file = "deeplearning.pdf"
question = input("You: ")

#we also give it an identity outside of functions
prompt = f"""Identity: Malik, you extract and answer only from the given {file}.
Workflow: 
1. First, use {read_file} to get the text from {file}.
2. Second, if the question requires ertain calculations or mathematical expression use {calculator}
3. Third, use {make_db_chunks} to chunk and store the text if needed.
4. Fourth, use {search} to access the  find the exact answer from the whole database if needed.
5. Finally, answer the user's question: {question}
Do not call the same tool twice in a row. If you already have the file contents in the chat history, do not call `read_file` again,
do not answer from your internal knowledge or any external source.
if the answer isn't mentioned just say i don't know, it wasn't mentioned"""

#init the model
load_dotenv()
os.getenv("MISTRAL_API_KEY")
model =init_chat_model( "mistral-small-latest",
     model_provider="mistralai",
    temperature=0.2)

messages = [prompt]
fn = [read_file, calculator, make_db_chunks, search]
mod_tools_bound = model.bind_tools(fn)

tool_map = {
    "read_file": read_file,
    "make_db_chunks": make_db_chunks,
    "search": search,
    "calculator": calculator
}

max_steps = 10
for steps in range(max_steps):
    response = mod_tools_bound.invoke(messages)

    if not response.tool_calls:
        print("\nFinal Answer:")
        print(response.content)
        break
    messages.append(response)

    for tool_desc in response.tool_calls:
        T_name = tool_desc["name"]
        T_arg = tool_desc["args"]
        T_id = tool_desc["id"]
        selected_tool = tool_map.get(T_name)
        if selected_tool:
            action = selected_tool.invoke(T_arg)
            print(action)
            messages.append(ToolMessage(content=str(action), tool_call_id=T_id))
        else:
            print(f"Tool {T_name} not found!")

    print(response.tool_calls)
else:
    print("reached max steps without a final answer, hence stopped the loop")