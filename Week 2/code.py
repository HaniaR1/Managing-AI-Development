import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
 
DATA_PATH = Path("data/winequality-red.csv")
 
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Place winequality-red.csv inside the data folder.")
    return pd.read_csv(path, sep=";")
 
def main():
    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Missing values:
", df.isna().sum())
 
    df["good_quality"] = (df["quality"] >= 7).astype(int)
    X = df.drop(columns=["quality", "good_quality"])
    y = df["good_quality"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }
 
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"
{name}")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("Precision:", precision_score(y_test, predictions, zero_division=0))
        print("Recall:", recall_score(y_test, predictions, zero_division=0))
        print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
        print("Confusion Matrix:
", confusion_matrix(y_test, predictions))
 
if __name__ == "__main__":
    main()
 
