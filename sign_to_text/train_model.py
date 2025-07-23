import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("A_data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(f"Accuracy on Test Data: {model.score(X_test, y_test)*100:.2f}%")

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
