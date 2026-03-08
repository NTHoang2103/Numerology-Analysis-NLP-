#1 Import thư viện
import streamlit as st
import torch
import joblib
from transformers import BertTokenizer
from model import BertClassifier

#2Load các model
# Load vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
# Load Logistic Regression
lr_model = joblib.load("models/logistic_regression_model.pkl")
# Load Random Forest
rf_model = joblib.load("models/random_forest_model.pkl")
# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
num_classes = 4
bert_model = BertClassifier(num_classes)
bert_model.load_state_dict(torch.load("models/bert_song_classifier.pth", map_location="cpu"))
bert_model.eval()

def predict_lr(text):
    X = vectorizer.transform([text])
    pred = lr_model.predict(X)[0]
    return pred

def predict_rf(text):
    X = vectorizer.transform([text])
    pred = rf_model.predict(X)[0]
    return pred

def predict_bert(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = bert_model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )
    pred = torch.argmax(outputs, dim=1).item()
    return pred

#Mapping class → genre
genres = {
    0: "Country",
    1: "Hip-Hop",
    2: "Electronic",
    3: "Rock"
}

st.title("🎵 Song Genre Classification")
st.write("Enter song lyrics and select a model to predict the genre.")
text = st.text_area("Enter lyrics")
model_choice = st.selectbox(
    "Select Model",
    ["BERT", "Logistic Regression", "Random Forest"]
)
if st.button("Predict"):

    if model_choice == "BERT":
        pred = predict_bert(text)

    elif model_choice == "Logistic Regression":
        pred = predict_lr(text)

    else:
        pred = predict_rf(text)
    st.success(f"Predicted Genre: {genres[pred]}")