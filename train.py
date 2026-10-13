import json
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import to_onnx

# 1. Define folder layout configurations
DATA = Path("data/drug200.csv")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 2. Load and segment input target matrices
df = pd.read_csv(DATA)
X = df.drop(columns=["Drug"])
y = df["Drug"]

categorical = ["Sex", "BP", "Cholesterol"]
numerical = ["Age", "Na_to_K"]

# 3. Build preprocessing and classification pipeline layout
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# 4. Split data partitions symmetrically
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Execute model training calculations
pipeline.fit(X_train, y_train)

# 6. Evaluate metrics array outputs
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

# 7. Save original Python backend model for reproducibility logs
joblib.dump(pipeline, MODEL_DIR / "drug_model.joblib")

# 8. CRITICAL STEP 13: Export pipeline configuration to portable ONNX format
# Uses a single-row representative sample layout containing original text feature dimensions
sample = X_train.iloc[:1]
onnx_model = to_onnx(
    pipeline,
    sample,
    target_opset=17,
    options={id(pipeline): {"zipmap": False}},
)

with open(MODEL_DIR / "drug_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model exported successfully for browser-side deployment.")

# 9. Write validation tracking matrices out to files
with open(RESULTS_DIR / "metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted F1-score: {f1:.4f}\n\n")
    f.write(classification_report(y_test, predictions))

# 10. Generate and save custom evaluation confusion matrix chart graphics
ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation="vertical")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("Training, evaluation, and ONNX serialization completed successfully.")
