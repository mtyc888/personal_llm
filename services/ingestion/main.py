from fastapi import FastAPI, HTTPException
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

app = FastAPI()

@app.get('/ingest')
def ingestData(filePath: str):
    index = filePath.find('.')
    if index != -1:
        extension = filePath[index + 1:]
        match extension:
            case 'txt':
                content = readTxt(filePath)

            case _:
                return f"File extension not supported."

def readTxt(filePath):
    try:
        with open(filePath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    
def chunk(content):
    """
        This function focuses on cutting the block of text into smaller pieces for the LLM

        Splits text into overlapping chunks, chunk by sentences
        overlap: number of characters shared between consecutive chunks
    """
    # we break down content from whole paragraphs of text to sentences list
    sentences = sent_tokenize(content)

    # then we make sure each sentences join with other sentences where it is < n (1000 characters) per sentence
    chunks = []
    current_chunk = ""
    n = 1000
    for sentence in sentences:
        # if i add the sentences, will it stil be under 500 characters?
        if len(current_chunk) + len(sentence) < n:
            # if it fits, append it to the current chunk
            current_chunk += " " + sentence
        else:
            # if it does not fit, append it to the chunks list
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def embedding(chunks):
    pass