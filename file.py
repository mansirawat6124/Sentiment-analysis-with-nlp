import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    "review": [
        "This product is amazing, I love it!",
        "Terrible experience, never buying again.",
        "Absolutely fantastic quality, very happy!",
        "Not good, totally disappointed.",
        "Best purchase I have ever made!",
        "Waste of money, do not recommend."
    ],
    "sentiment": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["cleaned_review"] = df["review"].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
