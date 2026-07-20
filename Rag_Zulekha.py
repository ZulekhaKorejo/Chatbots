#import
from pypdf import PdfReader
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from dotenv import load_dotenv
from mistralai.client import Mistral

#now open/read the file imported
reader = PdfReader("2K24_DS_104_Zulekha_final.pdf")

#clean up your pdf from blank spaces and layouts like fonts and so on
clean_text = ""
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if text.strip():
        clean_text += text + '\n'

#upload an encoder/embedder func at the same time to make it into vectors for embeddings
encode_embed = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")


#define how to chunk/split the words, apply it on the pdf
splitted = RecursiveCharacterTextSplitter(
    chunk_size = 500,       #how many characters in each chunk
    chunk_overlap = 50,     #overlap means how much characters the chunks share at th edges, this helps keeping the context in at least one cunk if its sliced mid sentence
)

chunks = splitted.split_text(clean_text)

#setting up an env for chroma database, where it'll load
Data_path = r"data"
Chroma_path = r"Chroma_Database"

#now get chroma client ready and the collections of the pdf
client = chromadb.PersistentClient(path=Data_path)
#how collections are uploaded in the db, emdedded, and metadata of how much close they r (cosine similarity)
#where hnsw:space means Hierarchical Navigable Small World used to search and organize vectors in the data
# without navigating each in the space, using the metrics cosine similarity hich is shown as "distances"
collection = client.get_or_create_collection(name="pdf_collection", embedding_function=encode_embed, metadata={"hnsw:space": "cosine"})

doc = []        #the imp info we'll load
meta = []       #file data and source
Ids = []        #keeps track of each chunk

i=0
for chunk in chunks:
    doc.append(chunk)       #i already made it str in my extraction so i won't do it again
    Ids.append(f"ID_{str(i)}")
    meta.append({
        "source_file": "2K24_DS_104_Zulekha_final.pdf",
        "index": i
    })
    i+=1

#now we add them to the db
results = collection.upsert(
    documents=doc,
    ids= Ids,
    metadatas= meta
)

#initialize the model
load_dotenv()
model = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=model)

#define the query
questions = ["What does digit classification utilize to get it's objective?",
             "what is MNIST dataset about?",
             "what are the evaluation metrics we use in MNIST?",
             "What is machine learning?",
             "What does decision tree do?",
             "What does regression do?",
             "what issues does Decision tree and KNN face?",
             "what are the parameters used in KNN?",
             "what do we do after defining the models?",
             "what two parameteres did we extract using model.items() in our code?"]
#question 4 and question 6 are unrelated, while question 7 is mentioned but not directly,
#  you can get it by reading comprehension yet it said "i dont know", so the result is 9/10 correct

for q, question in enumerate(questions):
    print(f"Query {q+1} -> {question}\n-------\n")
    #define the result by AI
    answer = collection.query(
        query_texts= question,
        n_results=1     #closest one answer
    )
    
    #now we extract the extract the needed info only from the doc, where [0][0] means the first/closest answer, otherwise it'll be empty -> no crash
    context = answer["documents"][0][0] if answer["documents"] and answer["documents"][0] else ""
    #make sure we extract the cosine too from each
    
    #also we extract the metadata too
    metadata = answer["metadatas"][0][0] if answer["metadatas"] and answer["metadatas"][0] else {}
    
    #get the prompt ready
    sys_prompt = f"""You are Klee, you extract the requested data from the provided file {context},
    don't go overboard and search anything outside of it, nor make anyting up by yourself,
    if you dont know the answer just say I Dont Know, it's not mentioned in the file
    answering the query: {question}"""

    # make it know how to answer, and differentiate between user and sys
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages= [
            {"role": "user", "content": sys_prompt}
        ]
    )

    print("\n Klee:", response.choices[0].message.content)

    time.sleep(0.5)

    if answer.get("distances") and answer["distances"][0]:
        cosine_dist = answer["distances"][0][0]         #calculate the distance and substract from 1 to see the range from 0-1
        cosine_simi = 1.0- cosine_dist
        print(f"cosine similarity: {cosine_simi:.2f}")
    else:
        print(f"no distance score found")


    print(f"\n metadata source information: {metadata} \n--------------\n")




#from my comparison using different chunk sizes, i realized it rudecued
# the cosine similarity in both cases (increasing and even decreasing the number way too much)
# in the original 500 chunks it was 0.67
# using 300 chunks it lowered to 0.62
# and lastly making it 800 just lowered it further to 0.57

#related question:
#  What does digit classification utilize to get it's objective?

#non related question: 
# What is machine learning?