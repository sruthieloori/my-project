Here’s a **clean, minimal README (only essential matter)** you can paste into GitHub:

````markdown
# Hate Speech Detection App

A Streamlit web app that classifies text as:
- Hate Speech
- Offensive Language
- Neutral

## Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install streamlit pandas scikit-learn nltk
````

## Run

```bash
streamlit run app.py
```

## How it works

* Preprocesses text (lowercase, remove symbols, stopwords, stemming)
* Converts text using TF-IDF
* Trains a LinearSVC model
* Predicts the label for user input

## Files

* `app.py` – main application
* `labeled_data.csv` – dataset
* `random_forest_model.pkl` – saved model (optional)

## Note

The model trains every time the app runs.

```
```
