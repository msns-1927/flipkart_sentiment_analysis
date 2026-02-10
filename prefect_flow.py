from prefect import flow, task
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

@task
def train_model(X_train, X_test, y_train,y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, f1_score(y_test, preds)

@task
def log_to_mlflow(model, f1):
    mlflow.set_experiment("Flipkart_Sentiment_Analysis")
    with mlflow.start_run(run_name="Prefect_RandomForest"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("orchestrator", "Prefect")
        mlflow.set_tag("pipeline", "Training + Logging")
        mlflow.sklearn.log_model(model, name="model")

@flow(name="Flipkart Sentiment Training Flow")
def training_flow(X_train, X_test, y_train, y_test):
    model, f1 = train_model(X_train, X_test, y_train, y_test)
    log_to_mlflow(model, f1)

if __name__ == "__main__":
    print("Running Prefect Flow...")

    # Load dataset
    df = pd.read_csv("data.csv")   # use your actual dataset path

    X = df["clean_review"]
    y = df["sentiment"]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    training_flow(X_train, X_test, y_train, y_test)