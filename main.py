import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the data
try:
    with open('data.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: intents.json file not found.")
    data = {"intents": []}

# Load trained model
try:
    model = keras.models.load_model('chat_model.keras')
except:
    print("Error: Failed to load chat_model.keras")
    model = None

# Load tokenizer object
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Error: tokenizer.pickle file not found.")
    tokenizer = None

# Load label encoder object
try:
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
except FileNotFoundError:
    print("Error: label_encoder.pickle file not found.")
    lbl_encoder = None

# Parameters
max_len = 20

# Regex pattern for tracking ID
pattern = r'[A-Z]{2}\d{9}'

class ChatInput(BaseModel):
    message: str

def extract_tracking_id(message):
    match = re.search(pattern, message)
    return match.group() if match else None

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "message": "The requested URL was not found on the server.",
            "available_routes": [
                "/",
                "/chat",
                "/debug"
            ]
        }
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Chatbot API. Send POST requests to /chat to interact with the bot."}

@app.options("/chat")
async def options_chat():
    return JSONResponse(status_code=204)

@app.post("/chat")
async def chat(chat_input: ChatInput):
    if not all([model, tokenizer, lbl_encoder]):
        raise HTTPException(status_code=500, detail="Server is not properly initialized. Check server logs.")
    
    inp = chat_input.message
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                         truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    confidence = np.max(result)
    
    if confidence > 0.8:
        for i in data['intents']:
            if i['tag'] == tag[0]:
                if tag[0] == 'location':
                    tracking_id = extract_tracking_id(inp)
                    if tracking_id:
                        response = f"Your tracking ID is {tracking_id}"
                    else:
                        response = "No valid tracking ID found in your message."
                else:
                    response = random.choice(i['responses'])
                break  # Stop the loop once the response is found

    else:
        response = "Sorry, I didn't understand that."

    return {"response": response}

@app.get("/debug")
async def debug():
    return {
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "label_encoder_loaded": lbl_encoder is not None,
        "intents_loaded": len(data['intents']) > 0 if 'intents' in data else False
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting server... Please wait until you see 'Application startup complete' message.")
    uvicorn.run(app)