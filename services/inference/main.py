from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama

app = FastAPI()

class GenerateReq(BaseModel):
    user_query: str
    data_list: list

@app.post("/generate")
def generate(request: GenerateReq):
    context = sorted(request.data_list, key=lambda x: x['distances'])
    context_text = "\n---\n".join(item['content'] for item in context[:5])
    question = request.user_query

    def stream_processor():
        response_stream = ollama.chat(
            model='phi4-mini',
            messages=[
                {
                    'role': 'system',
                    'content': 'You answer questions using ONLY the provided context. Rules: '
                            '1. Quote the exact section number from the context. '
                            '2. Do NOT paraphrase numbers, dates, or durations, copy them exactly. '
                            '3. If the answer is not directly stated, look for related information in the context and share it. '
                            '4. Only say "not covered" if the context has absolutely nothing related to the question. '
                            '5. Keep answers concise but complete, include all relevant details from the context.'
                },
                {
                    'role': 'user',
                    'content': f"Context:\n\n{context_text}\n\n---\nQuestion: {question}\n\nAnswer using ONLY the context above. Cite the section number."
                },
            ],
            stream=True
        )

        for chunk in response_stream:
            yield chunk.message.content

    return StreamingResponse(stream_processor(), media_type="text/plain")