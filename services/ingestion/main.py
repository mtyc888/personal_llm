from fastapi import FastAPI, HTTPException
import nltk
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import chromadb
import os
import uuid
from pydantic import BaseModel
import fitz
nltk.download('punkt_tab')


app = FastAPI()

persist_directory = './chroma_storage'
model = SentenceTransformer('all-MiniLM-L6-v2')
# initialize the client
chroma_client = chromadb.PersistentClient(path=persist_directory)
# get or create a collection
collection = chroma_client.get_or_create_collection(name="my_vault")

# pydantic models
class IngestRequest(BaseModel):
    filePath: str

class RetrieveRequest(BaseModel):
    query: str

@app.post('/ingest')
def ingestData(request: IngestRequest):
    filePath = request.filePath
    index = filePath.find('.')
    content = ""
    if index != -1:
        extension = filePath[index + 1:]
        match extension:
            case 'txt':
                content = readTxt(filePath)
            case 'pdf':
                content = readPdf(filePath)
            case _:
                return f"File extension not supported: {extension}."
    chunks = chunk(content)
    embed = embedding(chunks)
    storeInDB(embed, chunks, filePath) 
            
@app.post('/retrieve')
def retrieveData(request: RetrieveRequest):
    query = request.query
    embedding = model.encode(query).tolist()
    result = retrieveFromDB(embedding)

    return result
    
def readPdf(filePath):
    try:
        doc = fitz.open(filePath)
        text_content = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # extract text
            text_content += page.get_text() + "\n"

        return text_content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File: {filePath} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    
def readTxt(filePath):
    try:
        with open(filePath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    
def chunk(content, max_chars=1000, sentence_overlap=2):
    """
    Split text into overlapping chunks, preferring sentence boundaries.
    Overlap is done with whole sentences (not characters).
    """
    if not content.strip():
        return []

    sentences = sent_tokenize(content)
    if not sentences:
        # fallback if no sentences detected
        return [content[:max_chars].rsplit(" ", 1)[0]] if len(content) > max_chars else [content]

    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_len = 0
        start_i = i

        # greedily add sentences until we would exceed limit
        while i < len(sentences):
            sentence = sentences[i]
            # rough estimate including space
            added_len = len(sentence) + (1 if current_chunk else 0)

            if current_len + added_len > max_chars:
                break

            current_chunk.append(sentence)
            current_len += added_len
            i += 1

        # if we didn't take any sentence → force take at least one
        if not current_chunk and i < len(sentences):
            current_chunk = [sentences[i]]
            i += 1

        chunk_text = " ".join(current_chunk).strip()
        chunks.append(chunk_text)
        if sentence_overlap > 0:
            # slide window back by overlap sentences
            i = max(start_i + 1, i - sentence_overlap)

    return chunks

def embedding(chunks):
    embeddings = model.encode(chunks)
    return embeddings

def storeInDB(embeddings, chunks, filePath):
    uuid_code = uuid.uuid4()
    list_of_ids = [f"{os.path.basename(filePath)}-{uuid_code}-{i}" for i in range(len(chunks))]
    list_of_metadata = [{"source":filePath} for _ in range(len(chunks))]
    collection.add(
        ids=list_of_ids,
        embeddings= embeddings,
        documents = chunks,
        metadatas = list_of_metadata    
    )

def retrieveFromDB(embedding):
    results = collection.query(
        query_embeddings = [embedding],
        n_results = 3,
        include=['documents','distances','metadatas']
    )
    res = []
    for i in range(len(results['documents'][0])):
        res.append({
            "content":results['documents'][0][i],
            "distances":results['distances'][0][i],
            "metadatas":results['metadatas'][0][i]
        })
    return res