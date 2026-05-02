import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# --- MOVE THIS TO THE TOP ---
# This MUST be the first st. command the code sees
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬")
# 1. SETUP: Download tools
# (Note: st.cache_resource here is fine because it's a decorator, not a command execution)
@st.cache_resource
def download_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_data()

# 2. DEFINE THE MODEL ARCHITECTURE
class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.rnn(x) 
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out)

# 3. LOAD ASSETS
@st.cache_resource
def load_assets():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    model = SentimentRNN(input_size=5000)
    model.load_state_dict(torch.load("rnn_model.pth"))
    model.eval()
    return tfidf, model

tfidf, model = load_assets()

# 4. PREPROCESSING
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_user_input(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text) 
    text = re.sub(r"[^a-z\s]", "", text) 
    tokens = word_tokenize(text)
    cleaned = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)

# 5. USER INTERFACE
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type a movie review below to see if the AI thinks it is **Positive** or **Negative**.")

user_review = st.text_area("Enter Review:", placeholder="The movie was absolutely fantastic...")

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some text first!")
    else:
        processed_text = clean_user_input(user_review)
        vectorized_input = tfidf.transform([processed_text]).toarray()
        input_tensor = torch.tensor(vectorized_input).float().unsqueeze(1)
        
        with torch.no_grad():
            probability = model(input_tensor).item()
        
        if probability > 0.5:
            st.success(f"### Result: POSITIVE ✅")
            st.write(f"Confidence Score: {probability:.2%}")
        else:
            st.error(f"### Result: NEGATIVE ❌")
            st.write(f"Confidence Score: {1 - probability:.2%}")