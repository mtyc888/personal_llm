from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import nltk
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import chromadb
import os
import re
import uuid
from pydantic import BaseModel
import fitz
nltk.download('punkt_tab')

app = FastAPI()

persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_storage')
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection(name="my_vault")

class IngestRequest(BaseModel):
    filePath: str

class RetrieveRequest(BaseModel):
    query: str

@app.get('/sections')
def getSections():
    results = collection.get(include=['metadatas'])
    sections = set()
    for meta in results['metadatas']:
        if meta.get('section') and meta['section'] != 'General':
            sections.add(meta['section'])
    return {"sections": sorted(sections)}

@app.post('/ingest')
def ingestData(request: IngestRequest):
    filePath = request.filePath

    extension = os.path.splitext(filePath)[1].lower().replace('.', '')
    match extension:
        case 'txt':
            content = readTxt(filePath)
        case 'pdf':
            content = readPdf(filePath)
        case _:
            return JSONResponse(status_code=400, content={"error": f"File extension not supported: {extension}"})

    chunks = chunk(content)

    text_only_list = [c["text"] for c in chunks]

    metadatas = []
    for c in chunks:
        metadatas.append({
            "source": filePath,
            "page": c.get("page", 0),
            "section": c.get("section", "General")
        })

    embeds = embedding(text_only_list)
    storeInDB(embeds, text_only_list, metadatas, filePath)
    return JSONResponse(
        status_code=200,
        content={"message": "ingestion successful", "chunks": len(chunks)}
    )

@app.post('/retrieve')
def retrieveData(request: RetrieveRequest):
    query = request.query
    query_embedding = model.encode(query).tolist()
    result = retrieveFromDB(query_embedding)
    return result

def readPdf(filePath):
    try:
        doc = fitz.open(filePath)
        pdf_dict = {}
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_content = page.get_text()
            if text_content.strip():
                pdf_dict[page_num] = text_content
        return pdf_dict
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File: {filePath} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

def readTxt(filePath):
    try:
        txt_dict = {}
        with open(filePath, 'r', encoding='utf-8') as file:
            content = file.read()
            txt_dict[0] = content
        return txt_dict
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")


# section chunking

def detect_section_header(text):
    patterns = [
        r'^\d+\.\s+[A-Z]',           # '1. General Employment'
        r'^\d+\.\d+\s+[A-Z]',        # '3.2 Sick Leave'
        r'^\d+\.\d+\.\d+\s+[A-Z]',   # '3.2.1 Sub Section'
        r'^Section\s+\d+',             # 'Section 3'
        r'^SECTION\s+\d+',             # 'SECTION 3'
        r'^[A-Z][A-Z\s&,]{4,}$',      # 'COMPENSATION & BENEFITS'
    ]
    for pattern in patterns:
        if re.match(pattern, text.strip()):
            return True
    return False

def chunk(content, max_chars=800, sentence_overlap=2):
    if isinstance(content, dict):
        all_chunks = []
        current_section = "General"
        current_block = []
        last_page = 1

        for page_num, page_text in content.items():
            if not page_text or not page_text.strip():
                continue

            lines = page_text.split('\n')

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                if detect_section_header(stripped):
                    # Flush previous block with its section header
                    if current_block:
                        block_text = ' '.join(current_block)
                        sub_chunks = _chunk_text(block_text, max_chars, sentence_overlap)
                        for c in sub_chunks:
                            all_chunks.append({
                                "text": f"[{current_section}] {c}",
                                "page": last_page,
                                "section": current_section
                            })
                        current_block = []

                    current_section = stripped
                    # Keep header as first line of new block
                    current_block.append(stripped)
                else:
                    current_block.append(stripped)

                last_page = page_num + 1

        # Flush final block
        if current_block:
            block_text = ' '.join(current_block)
            sub_chunks = _chunk_text(block_text, max_chars, sentence_overlap)
            for c in sub_chunks:
                all_chunks.append({
                    "text": f"[{current_section}] {c}",
                    "page": last_page,
                    "section": current_section
                })

        return all_chunks

    else:
        if not content or not content.strip():
            return []
        chunks = _chunk_text(content, max_chars, sentence_overlap)
        return [{"text": c, "page": 0, "section": "General"} for c in chunks]

def _chunk_text(text, max_chars=800, sentence_overlap=2):
    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text)
    if not sentences:
        return [text[:max_chars]] if len(text) > max_chars else [text]

    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_len = 0
        start_i = i

        while i < len(sentences):
            sentence = sentences[i]
            added_len = len(sentence) + (1 if current_chunk else 0)
            if current_len + added_len > max_chars:
                break
            current_chunk.append(sentence)
            current_len += added_len
            i += 1

        if not current_chunk and i < len(sentences):
            current_chunk = [sentences[i]]
            i += 1

        chunk_text = ' '.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if sentence_overlap > 0:
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
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

def retrieveFromDB(embedding):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        include=['documents', 'distances', 'metadatas']
    )
    res = []
    for i in range(len(results['documents'][0])):
        res.append({
            "content": results['documents'][0][i],
            "distances": results['distances'][0][i],
            "metadatas": results['metadatas'][0][i]
        })
    return res