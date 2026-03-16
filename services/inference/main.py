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
    context = sorted(request.data_list, key=lambda x : x['distances'])
    context_text = "\n---\n".join(item['content'] for item in context[:3])
    question = request.user_query

    # generator function
    def stream_processor():
        response_stream = ollama.chat(
            model='llama3.2:1b',
            messages=[
            {
                'role': 'system',
                'content': 'You are a personal assistant. Answer ONLY using the provided context. '
                        'Do not add information that is not explicitly stated in the context. '
                        'If the context does not contain the answer, say "I don\'t have that information." '
                        'Keep answers concise and factual.'
            },
            {
                'role': 'user',
                'content': f"Context:\n{context_text}\n\nQuestion: {question}"
            },
            ],
            stream=True
        )

        for chunk in response_stream:
            # get only the content string
            yield chunk.message.content

    return StreamingResponse(stream_processor(), media_type="text/plain")



