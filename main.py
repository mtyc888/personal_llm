from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)
VAULT_PATH = os.environ.get("VAULT_PATH", "./vault")
INGESTION_URL = "http://127.0.0.1:8002"
INFERENCE_URL = "http://127.0.0.1:8001"

"""
    watch dog implementation, watching "vault" folder
"""
class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext not in ['.pdf', '.txt']:
            logger.info(f"Skipping unsupported file: {event.src_path}.")
            return

        # added delay for file to finish writing
        prev_size = -1
        for _ in range(10):
            time.sleep(1)
            curr_size = os.path.getsize(event.src_path)
            if curr_size == prev_size:
                break
            prev_size = curr_size
        logger.info(f"New file detected: {event.src_path}.")
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(f"{INGESTION_URL}/ingest", json={"filePath": os.path.abspath(event.src_path)})
                if response.status_code == 200:
                    logger.info(f"Ingested: {event.src_path}.")
                else:
                    logger.error(f"Ingestion failed: {response}.")
        except Exception as e:
            logger.error(f"Vault watcher error:{e}.")

observer = Observer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    #startup
    os.makedirs(VAULT_PATH, exist_ok=True)
    observer.schedule(MyHandler(), VAULT_PATH, recursive=False)
    observer.start()
    logger.info(f"Watching vault: {VAULT_PATH}.")
    yield
    #shutdown
    observer.stop()
    observer.join()

app = FastAPI(lifespan=lifespan)

# CORS middleware, allowing all traffic since it's only a demo
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AskReq(BaseModel):
    query: str

class IngestRequest(BaseModel):
    filePath: str
"""
    This returns all the file names in the vault folder
"""
@app.get('/api/vault/files')
async def getFiles():
    try:
        folder = Path(VAULT_PATH)
        supported = ['.pdf','.txt']
        files = []
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() in supported:
                files.append(item.name)
        return JSONResponse(status_code=200, content={"files":files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting the files from {VAULT_PATH}: {e}")
"""
    When user uploads a file on the frontend, it uploads the file into vault folder by writing into it.
"""
@app.post('/api/vault/upload')
async def upload(file: UploadFile = File(...)):
    try:
        dest = Path(VAULT_PATH) / file.filename
        with open (dest, "wb") as f:
            content = await file.read()
            f.write(content)
        return JSONResponse(content={"status": "uploaded", "file": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}.")

"""
    This endpoint would get the file name that was uploaded to vault and call the ingestion service.
"""
@app.post('/api/vault/ingest')
async def ingest(request: IngestRequest):
    try:
        file_path = request.filePath
        if not file_path:
            raise Exception('file path needed.')
        async with httpx.AsyncClient() as client:
            ingest_response = await client.post(f'{INGESTION_URL}/ingest', json={"filePath":file_path})
            if ingest_response.status_code == 200:
                return JSONResponse(content={"status": "ingested", "file": file_path})
            else:
                raise HTTPException(status_code=500, detail=f'Ingestion endpoint error.')
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")
"""
    This endpoint will take user's query, retrieve context from ingestion service, 
    then give the related context to inference service to query the LLM for an reply.
"""
@app.post('/api/chat/ask')
async def ask(request: AskReq):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query needed")

    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        retr_response = await client.post(
            f'{INGESTION_URL}/retrieve', json={"query": query}
        )
        retr_response.raise_for_status()
        response_data = retr_response.json()
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    async def stream_generator():
        try:
            async with client.stream(
                "POST",
                f"{INFERENCE_URL}/generate",
                json={"user_query": query, "data_list": response_data}
            ) as response:
                if response.status_code != 200:
                    yield f"Error: Upstream returned {response.status_code}"
                    return
                async for chunk in response.aiter_text():
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(stream_generator(), media_type="text/plain")