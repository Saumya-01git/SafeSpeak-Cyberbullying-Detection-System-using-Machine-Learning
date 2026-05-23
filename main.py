from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import load_and_prepare_data, train_models

# Load data
data, clean_text = load_and_prepare_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"], data["label"], test_size=0.2, random_state=42
)

# Train models
vectorizer, log_model, nb_model, svm_model, rf_model = train_models(data)

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Predictions
log_acc = accuracy_score(y_test, log_model.predict(X_test_vec))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test_vec))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_vec))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_vec))

# Print results
print("\nMODEL ACCURACY:")
print("Logistic Regression:", log_acc)
print("Naive Bayes:", nb_acc)
print("SVM:", svm_acc)
print("Random Forest:", rf_acc)


# ==============================
# Chatbot
# ==============================
print("\nChatbot started (type 'exit')")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])

    prediction = log_model.predict(vec)[0]

    if prediction == 1:
        print("Bot: 🚫 Cyberbullying")
    else:
        print("Bot: ✅ Normal")