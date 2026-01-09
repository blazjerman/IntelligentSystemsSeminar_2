import sys
import json

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB




def start(file_config):

    # Load config
    with open(file_config, "r") as f:
        config = json.load(f)

    # Load data
    dataset = fetch_ucirepo(id=config["dataset_id"])
    x = dataset.data.features
    y = dataset.data.targets[config["target"]]

    print("Loaded " + config_file + " config.")
    print("Running...")

    # Preprocess data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config["numerical_cols"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config["categorical_cols"]),
        ]
    )

    # Setup models
    models = {

        "Decision Tree": DecisionTreeClassifier(
            random_state=config["seed"],
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=10000,
            C=0.5,
            class_weight="balanced"
        ),

        "Naive Bayes NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB()
    }


    # Train cross-validate
    cv_strategy = StratifiedKFold(
        n_splits=config["cross_validation"]["n_splits"],
        shuffle=True,
        random_state=config["seed"]
    )

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", model)
        ])

        cv_results = cross_validate(
            pipeline,
            x,
            y,
            cv=cv_strategy,
            scoring=["accuracy", "f1_macro", "recall_macro"],
            n_jobs=-1
        )

        print("-------------------------------------------" +
            "\nModel: " + str(name) +
            "\nCross-validation results (" + str(config['cross_validation']['n_splits']) + "-fold):" +
            "\n  Accuracy: " + str(round(cv_results['test_accuracy'].mean(), 4)) + " +/- " + str(round(cv_results['test_accuracy'].std(), 4)) +
            "\n  Macro F1: " + str(round(cv_results['test_f1_macro'].mean(), 4)) +
            "\n  Macro Recall: " + str(round(cv_results['test_recall_macro'].mean(), 4))
        )



if __name__ == '__main__':

    config_file = "config.json"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    start(config_file)