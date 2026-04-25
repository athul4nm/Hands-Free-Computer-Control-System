import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# load dataset
df = pd.read_csv("hand_module/gesture_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf")

model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

joblib.dump(model, "models/gesture_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved.")