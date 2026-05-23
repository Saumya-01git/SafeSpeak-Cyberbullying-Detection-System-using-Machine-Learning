import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# ==============================
# LOAD + CLEAN DATA
# ==============================
def load_and_prepare_data():
    data = pd.read_csv("data/cyberbullying_tweets.csv")

    data["label"] = data["cyberbullying_type"].apply(
        lambda x: 0 if x == "not_cyberbullying" else 1
    )

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    data["clean_text"] = data["tweet_text"].apply(clean_text)

    return data, clean_text


# ==============================
# TRAIN MODELS
# ==============================
def train_models(data):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    X = vectorizer.fit_transform(data["clean_text"])
    y = data["label"]

    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    nb_model = MultinomialNB()
    svm_model = LinearSVC()
    rf_model = RandomForestClassifier(n_estimators=100)

    log_model.fit(X, y)
    nb_model.fit(X, y)
    svm_model.fit(X, y)
    rf_model.fit(X, y)

    return vectorizer, log_model, nb_model, svm_model, rf_model