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
import mysql.connector
from mysql.connector import Error
import requests

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
pattern =  r'[A-Z]{4}\d{4}'  #r'[A-Z]{2}\d{9}'

class ChatInput(BaseModel):
    message: str
    #userId: str
    

def extract_tracking_id(message):
    match = re.search(pattern, message)
    return match.group() if match else None


db_config = {
       'host':'localhost',
       'port':3306,
       'database':'location',
       'user':'root',
       'password':'Lakshan@1234'
   }

def fetch_data(query,params):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query,params)
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return result

    except Error as e:
        print(f"Error:{e}")
        return None

""" 
def read_data(id:int):
    query = "SELECT * FROM current_location WHERE LocationId = %s"
    params = (id,)
    data = fetch_data(query,params)
    if data:
        return data
    else:
        return {"error":"Data not found"}
"""   
 
def getlocation(tracking_id):
    response = ""        
        # Send userID to another server
    try:
        external_server_url = "http://ec2-13-53-200-229.eu-north-1.compute.amazonaws.com/api/message/track"  # Replace with actual server URL
        params_ = {'trackingId': tracking_id}
        external_response = requests.get(external_server_url, params=params_)
        response = external_response.json()
        #response = response_data_json['data']['LocationName']
        print(response,flush=True)

            # Check if the request was successful
    except requests.RequestException as e:
        print(f" - Error occurred while sending tracking id {tracking_id} to external server: {str(e)}")
      
    return response
    
    
    
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
    #uid = chat_input.userId
    #print(f"Received message from user {uid}: {inp}", flush=True)
    
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
                        response = getlocation(tracking_id) 
                    else:
                        response = {"message":"No valid tracking ID found in your message."  }
                else:
                    response = {"message":random.choice(i['responses'])}
                break  # Stop the loop once the response is found

    else:
        response = {"message":"Sorry, I didn't understand that. Please ask questions related to our service"}

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