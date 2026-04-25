import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("head_module/final_gesture_dataset.csv")

# 2️⃣ Separate features and label
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column (label)

# 3️⃣ Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Create SVM model
model = SVC(kernel='rbf')   # you can try 'linear' also

# 5️⃣ Train model
model.fit(X_train, y_train)

# 6️⃣ Test model
y_pred = model.predict(X_test)

# 7️⃣ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 8️⃣ Save model
joblib.dump(model, "models/gesture_svm_model.pkl")

print("\nModel saved as gesture_svm_model.pkl")
