from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import nltk
import pickle
from json import load, dump, JSONDecodeError
from difflib import get_close_matches

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

app = FastAPI()

# Load model, words, and classes
model = load_model('chatbot_model_best.keras')
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load intents
with open('data.json', encoding='utf-8') as file:
    intents = json.load(file)

def load_knowledge_base(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data: dict = load(file)
    except (FileNotFoundError, JSONDecodeError):
        data = {"questions": []}
    return data

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as file:
        dump(data, file, indent=2)

def find_best_match(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]
    return None

# Load knowledge base
knowledge_base = load_knowledge_base('intents.json')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    
    best_match = find_best_match(user_message, [q["question"] for q in knowledge_base["questions"]])
    
    if best_match:
        answer = get_answer_for_question(best_match, knowledge_base)
        return {"response": answer}
    else:
        intents_list = predict_class(user_message)
        response = get_response(intents_list, intents)
        
        if response:
            return {"response": response}
        else:
            return {"response": "I don't know the answer. Can you teach me?"}

class QuestionUpdateRequest(BaseModel):
    question: str
    answer: str

@app.post("/add_question")
async def add_question(request: QuestionUpdateRequest):
    data = request.dict()  # Convert the request to a dictionary
    question = data.get("question")
    answer = data.get("answer")

    if not question or not answer:
        raise HTTPException(status_code=400, detail="Both question and answer must be provided.")
    
    knowledge_base = load_knowledge_base('intents.json')
    knowledge_base["questions"].append({"question": question, "answer": answer})
    save_knowledge_base('intents.json', knowledge_base)
    
    return {"message": "Question added successfully"}

@app.post("/update_answer")
async def update_answer(request: QuestionUpdateRequest):
    data = request.dict()  # Convert the request to a dictionary
    question = data.get("question")
    new_answer = data.get("answer")

    if not question or not new_answer:
        raise HTTPException(status_code=400, detail="Both question and new answer must be provided.")
    
    knowledge_base = load_knowledge_base('intents.json')
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            q["answer"] = new_answer
            save_knowledge_base('intents.json', knowledge_base)
            return {"message": "Answer updated successfully"}
    
    raise HTTPException(status_code=404, detail="Question not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
