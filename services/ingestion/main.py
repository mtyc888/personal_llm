from fastapi import FastAPI, HTTPException
import nltk
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import chromadb
import os
import uuid
from pydantic import BaseModel
import fitz
nltk.download('punkt')


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

    content = ""

    extension = os.path.splitext(filePath)[1].lower().replace('.', '')
    match extension:
        case 'txt':
            content = readTxt(filePath)
        case 'pdf':
            content = readPdf(filePath)
        case _:
            return f"File extension not supported: {extension}."
    chunks = chunk(content)
    text_only_list = [c["text"] if isinstance(c, dict) else c for c in chunks]
    metadatas = []
    for c in chunks:
        m = {"source": filePath}
        if isinstance(c, dict):
            m["page"] = c["page"]
        metadatas.append(m)

    embeds = embedding(text_only_list)
    storeInDB(embeds, text_only_list, metadatas, filePath) 
            
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
        pdf_dict = {}
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # extract text
            text_content = page.get_text()
            pdf_dict[page_num] = text_content

        return pdf_dict
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

    # handle pdf case
    if isinstance(content, dict):
        all_chunks_with_metadata = []
        all_chunk = []
        for page_num, page_text in content.items():
            if not page_text or not page_text.strip():
                continue
            page_chunks = chunk(page_text, max_chars, sentence_overlap)
            for c in page_chunks:
                all_chunks_with_metadata.append({
                    "text": c,
                    "page": page_num + 1
                })

        return all_chunks_with_metadata
    # handle txt case
    if not content or not content.strip():
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

def storeInDB(embeddings, chunks, metadatas, filePath):
    uuid_code = uuid.uuid4()
    list_of_ids = [f"{os.path.basename(filePath)}-{uuid_code}-{i}" for i in range(len(chunks))]
    collection.add(
        ids=list_of_ids,
        embeddings= embeddings,
        documents = chunks,
        metadatas = metadatas    
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