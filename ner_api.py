from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Define a request model
class NERRequest(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./xlm-roberta-ner")
model = AutoModelForTokenClassification.from_pretrained("./xlm-roberta-ner")

@app.post("/ner")
def get_ner(request: NERRequest):
    # Tokenize and evaluate the sentence
    inputs = tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Process the results
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    new_tokens, new_labels = [], []
    for token, prediction in zip(tokens, predictions[0].tolist()):
        if token.startswith("▁"):
            label = model.config.id2label[prediction]
            new_labels.append(label)
            new_tokens.append(token[1:])
        elif new_tokens:
            new_tokens[-1] = new_tokens[-1] + token

    return {"tokens": new_tokens, "labels": new_labels}
