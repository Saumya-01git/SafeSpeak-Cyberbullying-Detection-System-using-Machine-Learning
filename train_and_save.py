import joblib
from model import load_and_prepare_data, train_models

print("🚀 Training models... please wait")

data, clean_text = load_and_prepare_data()

vectorizer, log_model, nb_model, svm_model, rf_model = train_models(data)

joblib.dump(
    (data, vectorizer, log_model, nb_model, svm_model, rf_model),
    "saved_models.pkl"
)

print("✅ Models saved successfully!")