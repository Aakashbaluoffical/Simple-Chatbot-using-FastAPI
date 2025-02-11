from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

# Initialize FastAPI app
app = FastAPI(title="Chatbot API", description="A simple NLP-based chatbot using FastAPI.", version="1.0")

# Load model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Request and response schemas
class ChatRequest(BaseModel):
    messages: List[str]  # List of messages (conversation history)

class ChatResponse(BaseModel):
    response: str  # Bot's response

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API! Use the /chat endpoint to interact."}

@app.post("/chat", response_model=ChatResponse)
def chat_with_bot(request: ChatRequest):
    try:
        # Prepare the input for the model
        print("before ,",request.messages)
        conversation_history = "\n".join(request.messages)
        print("request :",conversation_history)
        input_ids = tokenizer.encode(conversation_history + tokenizer.eos_token, return_tensors="pt")

        # Generate a response
        response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        print("response_idsresponse_ids",response_ids)
        print("input_ids",input_ids)
        response_text = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
