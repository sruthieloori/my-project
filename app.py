import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text cleaning
def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)


# Load dataset
df = pd.read_csv("labeled_data.csv")

df["labels"] = df["class"].map({
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral"
})

df["clean_tweet"] = df["tweet"].apply(preprocess_text)

X = df["clean_tweet"]
y = df["labels"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LinearSVC()
model.fit(X_train, y_train)


# Prediction
def predict(text):

    clean = preprocess_text(text)

    vector = vectorizer.transform([clean])

    result = model.predict(vector)

    return result[0]


# Streamlit UI
st.title("Hate Speech Detection")

st.write("Enter a sentence to check if it is Hate Speech")

user_input = st.text_area("Enter Text")

if st.button("Predict"):

    if user_input == "":
        st.write("Please enter text")

    else:

        prediction = predict(user_input)

        st.write("Prediction:", prediction)