import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv("final_training_data.csv")
X = df.iloc[:, 0:63]
y = df.iloc[:, -1].astype(str).str.strip().str.title()

print("Label distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Evaluation Report:\n", classification_report(y_test, y_pred))

print(f"Accuracy on Test Data: {model.score(X_test, y_test)*100:.2f}%")

with open("sign_to_text\gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to sign_to_text/gesture_model.pkl")