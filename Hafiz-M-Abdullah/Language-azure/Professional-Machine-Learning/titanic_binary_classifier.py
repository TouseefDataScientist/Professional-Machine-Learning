import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
TARGET_COLUMN = "Survived"
MODEL_FILE = "titanic_random_forest.joblib"


def load_dataset(url: str) -> pd.DataFrame:
    """Load the public Titanic CSV dataset from GitHub."""
    df = pd.read_csv(url)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def prepare_features_and_target(df: pd.DataFrame):
    """Drop unnecessary columns and split into train/test sets."""
    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    X = df.drop(columns=drop_columns + [TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_model_pipeline() -> Pipeline:
    """Create a preprocessing + model pipeline for binary classification."""
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]
    categorical_features = [
        "Sex",
        "Embarked",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return model


def train_model(model: Pipeline, X_train, y_train):
    """Train the binary classification model."""
    model.fit(X_train, y_train)
    print("Model training complete")
    return model


def evaluate_model(model: Pipeline, X_test, y_test):
    """Evaluate the trained model on the holdout test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")

    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def main():
    df = load_dataset(DATA_URL)
    X_train, X_test, y_train, y_test = prepare_features_and_target(df)

    model = build_model_pipeline()
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    dump(model, MODEL_FILE)
    print(f"\nSaved trained model to {MODEL_FILE}")
    print(f"Final test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
