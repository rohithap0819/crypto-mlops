from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.train_baselines_v2 import (
    CLASSIFICATION_TARGET,
    REGRESSION_TARGETS,
    get_classification_models,
    get_regression_models,
    limit_training_rows,
    load_data,
    load_feature_columns,
    prepare_target_dataset,
)


RESULTS_FILE = Path(
    "reports/baseline_results_v2.csv"
)

DIAGNOSTICS_FILE = Path(
    "reports/baseline_diagnostics_v2.csv"
)

CLASS_DISTRIBUTION_FILE = Path(
    "reports/class_distribution_v2.csv"
)


def bps(value):
    return value * 10_000


def regression_diagnosis(
    target,
    best_row,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    zero_predictions = np.zeros(
        len(y_validation)
    )

    zero_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            zero_predictions,
        )
    )

    zero_mae = mean_absolute_error(
        y_validation,
        zero_predictions,
    )

    zero_r2 = r2_score(
        y_validation,
        zero_predictions,
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            predictions,
        )
    )

    model_mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    model_r2 = r2_score(
        y_validation,
        predictions,
    )

    actual_direction = np.sign(
        y_validation.to_numpy()
    )

    predicted_direction = np.sign(
        predictions
    )

    directional_accuracy = np.mean(
        actual_direction
        == predicted_direction
    )

    rmse_improvement = (
        (zero_rmse - model_rmse)
        / zero_rmse
        if zero_rmse != 0
        else 0
    )

    mae_improvement = (
        (zero_mae - model_mae)
        / zero_mae
        if zero_mae != 0
        else 0
    )

    print()
    print("=" * 70)
    print(
        f"V2 REGRESSION DIAGNOSIS: {target}"
    )
    print("=" * 70)

    print(
        f"Target mean: {y_validation.mean():.8f}"
    )
    print(
        f"Target std : {y_validation.std():.8f}"
    )

    print()
    print("ZERO-RETURN BASELINE")
    print(
        f"RMSE: {zero_rmse:.8f} "
        f"({bps(zero_rmse):.2f} bps)"
    )
    print(
        f"MAE : {zero_mae:.8f} "
        f"({bps(zero_mae):.2f} bps)"
    )
    print(
        f"R2  : {zero_r2:.6f}"
    )

    print()
    print(
        f"V2 BEST MODEL: {best_row['model']}"
    )
    print(
        f"RMSE: {model_rmse:.8f} "
        f"({bps(model_rmse):.2f} bps)"
    )
    print(
        f"MAE : {model_mae:.8f} "
        f"({bps(model_mae):.2f} bps)"
    )
    print(
        f"R2  : {model_r2:.6f}"
    )

    print()
    print("V2 VS ZERO-RETURN BASELINE")
    print(
        f"RMSE improvement: "
        f"{rmse_improvement * 100:.4f}%"
    )
    print(
        f"MAE improvement : "
        f"{mae_improvement * 100:.4f}%"
    )

    print()
    print(
        f"Directional accuracy: "
        f"{directional_accuracy * 100:.4f}%"
    )

    return {
        "task": "regression",
        "target": target,
        "best_model": best_row["model"],
        "target_mean": y_validation.mean(),
        "target_std": y_validation.std(),
        "zero_rmse": zero_rmse,
        "zero_mae": zero_mae,
        "zero_r2": zero_r2,
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "model_r2": model_r2,
        "rmse_improvement_vs_zero": rmse_improvement,
        "mae_improvement_vs_zero": mae_improvement,
        "directional_accuracy": directional_accuracy,
    }


def classification_diagnosis(
    best_row,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    classes = sorted(
        y_train.unique()
    )

    class_to_id = {
        name: index
        for index, name in enumerate(classes)
    }

    y_train_encoded = y_train.map(
        class_to_id
    )

    y_validation_encoded = (
        y_validation.map(class_to_id)
    )

    model.fit(
        X_train,
        y_train_encoded,
    )

    predictions = model.predict(
        X_validation
    )

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        X_train,
        y_train_encoded,
    )

    dummy_predictions = dummy.predict(
        X_validation
    )

    model_accuracy = accuracy_score(
        y_validation_encoded,
        predictions,
    )

    model_macro_f1 = f1_score(
        y_validation_encoded,
        predictions,
        average="macro",
        zero_division=0,
    )

    dummy_accuracy = accuracy_score(
        y_validation_encoded,
        dummy_predictions,
    )

    dummy_macro_f1 = f1_score(
        y_validation_encoded,
        dummy_predictions,
        average="macro",
        zero_division=0,
    )

    distribution = (
        y_validation
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"]
        / distribution["count"].sum()
        * 100
    )

    print()
    print("=" * 70)
    print("V2 CLASSIFICATION DIAGNOSIS")
    print("=" * 70)

    print()
    print("VALIDATION DISTRIBUTION")
    print(
        distribution.to_string(
            index=False
        )
    )

    print()
    print("MAJORITY BASELINE")
    print(
        f"Accuracy : "
        f"{dummy_accuracy:.6f}"
    )
    print(
        f"Macro-F1 : "
        f"{dummy_macro_f1:.6f}"
    )

    print()
    print(
        f"V2 BEST MODEL: "
        f"{best_row['model']}"
    )
    print(
        f"Accuracy : "
        f"{model_accuracy:.6f}"
    )
    print(
        f"Macro-F1 : "
        f"{model_macro_f1:.6f}"
    )

    print()
    print("V2 VS MAJORITY BASELINE")
    print(
        f"Accuracy change: "
        f"{model_accuracy - dummy_accuracy:+.6f}"
    )
    print(
        f"Macro-F1 change: "
        f"{model_macro_f1 - dummy_macro_f1:+.6f}"
    )

    return distribution, {
        "task": "classification",
        "target": CLASSIFICATION_TARGET,
        "best_model": best_row["model"],
        "majority_accuracy": dummy_accuracy,
        "majority_macro_f1": dummy_macro_f1,
        "model_accuracy": model_accuracy,
        "model_macro_f1": model_macro_f1,
        "accuracy_change_vs_majority": (
            model_accuracy
            - dummy_accuracy
        ),
        "macro_f1_change_vs_majority": (
            model_macro_f1
            - dummy_macro_f1
        ),
    }


def main():

    print("=" * 70)
    print("CRYPTO MLOPS - V2 BASELINE DIAGNOSTICS")
    print("=" * 70)

    results_df = pd.read_csv(
        RESULTS_FILE
    )

    train_df, validation_df = load_data()

    train_df = limit_training_rows(
        train_df
    )

    feature_columns = (
        load_feature_columns()
    )

    diagnostics = []
    distributions = []

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    for target in REGRESSION_TARGETS:

        subset = results_df[
            (results_df["task"] == "regression")
            & (results_df["target"] == target)
        ].sort_values("rmse")

        best_row = subset.iloc[0]

        X_train, y_train = (
            prepare_target_dataset(
                train_df,
                target,
                feature_columns,
            )
        )

        X_validation, y_validation = (
            prepare_target_dataset(
                validation_df,
                target,
                feature_columns,
            )
        )

        model = get_regression_models()[
            best_row["model"]
        ]

        diagnostic = regression_diagnosis(
            target=target,
            best_row=best_row,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        diagnostics.append(
            diagnostic
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    subset = results_df[
        results_df["task"]
        == "classification"
    ].sort_values(
        "macro_f1",
        ascending=False,
    )

    best_row = subset.iloc[0]

    X_train, y_train = (
        prepare_target_dataset(
            train_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    X_validation, y_validation = (
        prepare_target_dataset(
            validation_df,
            CLASSIFICATION_TARGET,
            feature_columns,
        )
    )

    model = get_classification_models()[
        best_row["model"]
    ]

    distribution, diagnostic = (
        classification_diagnosis(
            best_row=best_row,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
        )
    )

    diagnostics.append(
        diagnostic
    )

    distribution[
        "target"
    ] = CLASSIFICATION_TARGET

    distributions.append(
        distribution
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pd.DataFrame(
        diagnostics
    ).to_csv(
        DIAGNOSTICS_FILE,
        index=False,
    )

    pd.concat(
        distributions,
        ignore_index=True,
    ).to_csv(
        CLASS_DISTRIBUTION_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("V2 DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {DIAGNOSTICS_FILE}"
    )

    print(
        f"Saved: "
        f"{CLASS_DISTRIBUTION_FILE}"
    )


if __name__ == "__main__":
    main()