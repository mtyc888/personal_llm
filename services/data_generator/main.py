from fastapi import FastAPI, HTTPException
from pathlib import Path
import requests
import json
import re 
import fitz
from json_repair import repair_json

app = FastAPI()

@app.post('/generate-datasets')
def generateDataSets():
    output_file = 'my_training_data.jsonl'
    p = Path('C:\\Users\\HW000\\OneDrive\\Desktop\\personal_llm\\vault')
    file_list = [str(path.resolve()) for path in p.iterdir() if path.is_file() and path.suffix in ['.pdf','.txt']]
    print(f'file list {file_list}')
    for file in file_list:
        if ".pdf" in file:
            data_dict = readPdf(file)
        else:
            data_dict = readTxt(file)

        for page_num, content in data_dict.items():
            if not content.strip():
                continue
            
            query = (
                f"I will provide you with text from a document. Generate 5 high-quality "
                f"instruction-response pairs based ON ONLY this text. "
                f"Format your output ONLY as a JSON list of objects like this: "
                f'[{{ "instruction": "...", "input": "", "output": "..." }}].\n\n'
                f"TEXT:\n{content}"
            )

            # call inference service
            response = requests.post(
                "http://127.0.0.1:8000/generate-data",
                json={"user_query": query, "data_list": []}
            )
            
            if response.status_code == 200:
                try:
                    # parse the string output from the LLM into actual json
                    raw_output = response.json()
                    print(f"raw_output: {raw_output}")
                    json_string = clean_data(raw_output)
                    qa_pairs = repair_json(json_string, return_objects=True)

                    with open(output_file, "a", encoding="utf-8") as file:
                        for pair in qa_pairs:
                            file.write(json.dumps(pair) + "\n")
                except Exception as e:
                    print(f"Error parsing LLM output on page {page_num}: {e}")
    return {"status":"success", "file":output_file}
    
def clean_data(raw_content):
    match = re.search(r'(\[.*\])', raw_content, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str.replace("'", '"')
    return raw_content 

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
        txt_dict = {}
        with open(filePath, 'r', encoding='utf-8') as file:
            content = file.read()
            txt_dict[0] = content
        return txt_dict
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")