import streamlit as st
import pandas as pd
import re
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# PAGE CONFIG + UI
# ======================
st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    color: #222;
    font-family: 'Segoe UI', sans-serif;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #434343, #000000);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* TITLE */
h1 {
    color: #222 !important;
    text-align: center;
    font-size: 42px !important;
    font-weight: bold;
}

/* SUBHEADINGS */
h2, h3 {
    color: #333 !important;
}

/* INPUT BOX */
.stTextInput input {
    border-radius: 12px !important;
    border: 2px solid #ff7e5f !important;
    padding: 10px !important;
    background-color: white !important;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] {
    border-radius: 12px !important;
    border: 2px solid #ff7e5f !important;
    background-color: white !important;
}

/* BUTTON */
div.stButton > button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border: none;
    border-radius: 14px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s ease;
}

div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #dd2476, #ff512f);
}

/* ALERT BOXES */
div[data-testid="stAlert"] {
    border-radius: 14px;
    font-size: 16px;
    font-weight: bold;
}

/* SUCCESS */
div[data-testid="stAlert"]:has(svg[aria-label="success"]) {
    background-color: #d4edda !important;
    border-left: 8px solid green !important;
}

/* ERROR */
div[data-testid="stAlert"]:has(svg[aria-label="error"]) {
    background-color: #f8d7da !important;
    border-left: 8px solid red !important;
}

/* WARNING */
div[data-testid="stAlert"]:has(svg[aria-label="warning"]) {
    background-color: #fff3cd !important;
    border-left: 8px solid orange !important;
}

/* INFO */
div[data-testid="stAlert"]:has(svg[aria-label="info"]) {
    background-color: #d1ecf1 !important;
    border-left: 8px solid #17a2b8 !important;
}

/* CHART SPACING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* METRIC STYLE */
.css-1xarl3l {
    background-color: white;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "saved_models.pkl")

# ======================
# LOAD OR TRAIN MODEL
# ======================

if os.path.exists(model_path):

    data, vectorizer, log_model, nb_model, svm_model, rf_model = joblib.load(model_path)

else:
    from model import load_and_prepare_data, train_models

    data, clean_text = load_and_prepare_data()

    vectorizer, log_model, nb_model, svm_model, rf_model = train_models(data)

    joblib.dump(
        (data, vectorizer, log_model, nb_model, svm_model, rf_model),
        model_path
    )

    

# ======================
# CLEAN FUNCTION
# ======================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ======================
# TITLE
# ======================
# ======================
# SIDEBAR
# ======================

st.sidebar.title("🛡️ SafeSpeak")

page = st.sidebar.radio(
    "Go To",
    ["Home", "Model Comparison", "About"]
)

if page == "Home" :
    st.title("🛡️ Cyberbullying Detection System")

    # ======================
    # MODEL SELECTION
    # ======================
    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"]
    )

    # ======================
    # INPUT
    # ======================
    user_input = st.text_input("Enter message:")

    # ======================
    # ANALYZE
    # ======================
    if st.button("Analyze"):

        if user_input.strip() != "":

            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            # ----------- MODEL PREDICTION -----------
            if model_choice == "Logistic Regression":
                pred = log_model.predict(vec)[0]
                prob = log_model.predict_proba(vec)[0]
                conf = max(prob) * 100

            elif model_choice == "Naive Bayes":
                pred = nb_model.predict(vec)[0]
                prob = nb_model.predict_proba(vec)[0]
                conf = max(prob) * 100

            elif model_choice == "SVM":
                pred = svm_model.predict(vec)[0]
                conf = "N/A"

            else:
                pred = rf_model.predict(vec)[0]
                prob = rf_model.predict_proba(vec)[0]
                conf = max(prob) * 100

            # ----------- SAFE WORDS (avoid false positives) -----------
            safe_words = ["good", "nice", "great", "well done", "awesome"]
            if any(word in cleaned for word in safe_words):
                pred = 0

            # ----------- STRONG BULLYING KEYWORDS -----------
            bad_words = [
                "stupid", "idiot", "dumb", "hate", "loser", "ugly", "kill",
                "worthless", "nobody cares", "failure", "useless", "trash"
            ]

            found_bad = [word for word in bad_words if word in cleaned]

            # 👉 KEY FIX: force bullying if strong words found
            if found_bad:
                pred = 1

            # ----------- OUTPUT -----------
            if pred == 1:
                st.error("🚫 Cyberbullying detected")
            else:
                st.success("✅ Normal message")

            st.write("Confidence:", conf)

            # ----------- EXPLANATION -----------
            if found_bad:
                st.warning(f"⚠️ Offensive words detected: {found_bad}")
            else:
                st.info("No offensive keywords found")

# ======================
# MODEL COMPARISON
# ======================

if page == "Model Comparison":
    st.subheader("📊 Model Comparison")

    X = vectorizer.transform(data["clean_text"])
    y = data["label"]

    models = {
        "Logistic": log_model,
        "Naive Bayes": nb_model,
        "SVM": svm_model,
        "Random Forest": rf_model
    }

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X)

        results[name] = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred)
        }

    st.write(results)

    # ======================
    # ACCURACY GRAPH
    # ======================
    st.subheader("📊 Accuracy Comparison")

    accuracy_scores = {name: results[name]["Accuracy"] for name in results}
    st.bar_chart(accuracy_scores)

    # ======================
    # F1 GRAPH
    # ======================
    st.subheader("📊 F1 Score Comparison")

    f1_scores = {name: results[name]["F1 Score"] for name in results}
    st.bar_chart(f1_scores)

if page == "About":

    st.title("📘 About Project")

    st.write(
        '''
        SafeSpeak is a Machine Learning based Cyberbullying Detection System.

        Features:
        - Detects bullying text
        - Multiple ML models
        - Accuracy comparison
        - Interactive Streamlit UI

        Technologies Used:
        - Python
        - Streamlit
        - Scikit-learn
        - NLP
        '''
    )
