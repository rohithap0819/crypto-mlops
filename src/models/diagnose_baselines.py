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
from sklearn.linear_model import LogisticRegression

from src.models.train_baselines import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    REGRESSION_TARGETS,
    get_classification_models,
    get_regression_models,
    limit_training_rows,
    load_data,
    prepare_target_dataset,
)


# ============================================================
# PATHS
# ============================================================

RESULTS_FILE = Path("reports/baseline_results.csv")

DIAGNOSTICS_FILE = Path(
    "reports/baseline_diagnostics.csv"
)

CLASS_DISTRIBUTION_FILE = Path(
    "reports/class_distribution.csv"
)


# ============================================================
# HELPERS
# ============================================================

def percentage(value):
    return f"{value * 100:.4f}%"


def basis_points(value):
    return value * 10_000


def print_regression_diagnosis(
    target,
    y_validation,
    model_name,
    model_metrics,
    predictions,
):
    print()
    print("=" * 70)
    print(f"REGRESSION DIAGNOSIS: {target}")
    print("=" * 70)

    print(f"Target mean       : {y_validation.mean():.8f}")
    print(f"Target std        : {y_validation.std():.8f}")
    print(f"Target min        : {y_validation.min():.8f}")
    print(f"Target max        : {y_validation.max():.8f}")

    print()
    print("Target scale:")
    print(
        f"Mean              : "
        f"{percentage(y_validation.mean())}"
    )
    print(
        f"Std               : "
        f"{percentage(y_validation.std())}"
    )

    # --------------------------------------------------------
    # Naive zero-return baseline
    # --------------------------------------------------------

    zero_predictions = np.zeros(len(y_validation))

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

    print()
    print("NAIVE BASELINE: PREDICT ZERO RETURN")
    print(
        f"RMSE              : {zero_rmse:.8f} "
        f"({basis_points(zero_rmse):.2f} bps)"
    )
    print(
        f"MAE               : {zero_mae:.8f} "
        f"({basis_points(zero_mae):.2f} bps)"
    )
    print(
        f"R2                : {zero_r2:.6f}"
    )

    # --------------------------------------------------------
    # ML model
    # --------------------------------------------------------

    model_rmse = model_metrics["rmse"]
    model_mae = model_metrics["mae"]
    model_r2 = model_metrics["r2"]

    print()
    print(f"BEST ML MODEL: {model_name}")
    print(
        f"RMSE              : {model_rmse:.8f} "
        f"({basis_points(model_rmse):.2f} bps)"
    )
    print(
        f"MAE               : {model_mae:.8f} "
        f"({basis_points(model_mae):.2f} bps)"
    )
    print(
        f"R2                : {model_r2:.6f}"
    )

    # --------------------------------------------------------
    # Improvement versus naive
    # --------------------------------------------------------

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
    print("MODEL VS ZERO-RETURN BASELINE")
    print(
        f"RMSE improvement  : "
        f"{percentage(rmse_improvement)}"
    )
    print(
        f"MAE improvement   : "
        f"{percentage(mae_improvement)}"
    )

    # --------------------------------------------------------
    # Directional accuracy
    # --------------------------------------------------------

    actual_direction = np.sign(
        y_validation.to_numpy()
    )

    predicted_direction = np.sign(
        predictions
    )

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    )

    non_zero_mask = actual_direction != 0

    if non_zero_mask.any():
        non_zero_directional_accuracy = np.mean(
            predicted_direction[non_zero_mask]
            == actual_direction[non_zero_mask]
        )
    else:
        non_zero_directional_accuracy = np.nan

    print()
    print("DIRECTIONAL DIAGNOSIS")
    print(
        f"Directional accuracy: "
        f"{percentage(directional_accuracy)}"
    )
    print(
        f"Excluding zero actuals: "
        f"{percentage(non_zero_directional_accuracy)}"
    )

    return {
        "task": "regression",
        "target": target,
        "best_model": model_name,
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
        "non_zero_directional_accuracy": (
            non_zero_directional_accuracy
        ),
    }


def print_classification_diagnosis(
    y_validation,
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation_encoded,
):
    print()
    print("=" * 70)
    print(
        f"CLASSIFICATION DIAGNOSIS: "
        f"{CLASSIFICATION_TARGET}"
    )
    print("=" * 70)

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
    print("VALIDATION CLASS DISTRIBUTION")
    print(distribution.to_string(index=False))

    # --------------------------------------------------------
    # Majority class baseline
    # --------------------------------------------------------

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        X_train,
        y_train,
    )

    dummy_predictions = dummy.predict(
        X_validation
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

    # --------------------------------------------------------
    # Actual model
    # --------------------------------------------------------

    model_predictions = model.predict(
        X_validation
    )

    model_accuracy = accuracy_score(
        y_validation_encoded,
        model_predictions,
    )

    model_macro_f1 = f1_score(
        y_validation_encoded,
        model_predictions,
        average="macro",
        zero_division=0,
    )

    print()
    print("MAJORITY-CLASS BASELINE")
    print(
        f"Accuracy          : "
        f"{dummy_accuracy:.6f}"
    )
    print(
        f"Macro-F1          : "
        f"{dummy_macro_f1:.6f}"
    )

    print()
    print(f"BEST ML MODEL: {model_name}")
    print(
        f"Accuracy          : "
        f"{model_accuracy:.6f}"
    )
    print(
        f"Macro-F1          : "
        f"{model_macro_f1:.6f}"
    )

    print()
    print("MODEL VS MAJORITY BASELINE")
    print(
        f"Accuracy change   : "
        f"{model_accuracy - dummy_accuracy:+.6f}"
    )
    print(
        f"Macro-F1 change   : "
        f"{model_macro_f1 - dummy_macro_f1:+.6f}"
    )

    return distribution, {
        "task": "classification",
        "target": CLASSIFICATION_TARGET,
        "best_model": model_name,
        "majority_accuracy": dummy_accuracy,
        "majority_macro_f1": dummy_macro_f1,
        "model_accuracy": model_accuracy,
        "model_macro_f1": model_macro_f1,
        "accuracy_change_vs_majority": (
            model_accuracy - dummy_accuracy
        ),
        "macro_f1_change_vs_majority": (
            model_macro_f1 - dummy_macro_f1
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CRYPTO MLOPS - BASELINE DIAGNOSTICS")
    print("=" * 70)

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Baseline results not found: "
            f"{RESULTS_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, validation_df = load_data()

    train_df = limit_training_rows(
        train_df
    )

    results_df = pd.read_csv(
        RESULTS_FILE
    )

    diagnostics = []
    all_class_distributions = []

    # --------------------------------------------------------
    # Regression diagnostics
    # --------------------------------------------------------

    for target in REGRESSION_TARGETS:

        result = results_df[
            (results_df["task"] == "regression")
            & (results_df["target"] == target)
        ].sort_values("rmse")

        if result.empty:
            raise ValueError(
                f"No baseline result found for {target}"
            )

        best = result.iloc[0]
        best_model_name = best["model"]

        X_train, y_train = (
            prepare_target_dataset(
                train_df,
                target,
            )
        )

        X_validation, y_validation = (
            prepare_target_dataset(
                validation_df,
                target,
            )
        )

        models = get_regression_models()

        if best_model_name not in models:
            raise ValueError(
                f"Unsupported regression model: "
                f"{best_model_name}"
            )

        model = models[
            best_model_name
        ]

        print()
        print(
            f"Refitting {best_model_name} "
            f"for {target} diagnostics..."
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        model_metrics = {
            "rmse": float(best["rmse"]),
            "mae": float(best["mae"]),
            "r2": float(best["r2"]),
        }

        diagnostic = (
            print_regression_diagnosis(
                target=target,
                y_validation=y_validation,
                model_name=best_model_name,
                model_metrics=model_metrics,
                predictions=predictions,
            )
        )

        diagnostics.append(diagnostic)

    # --------------------------------------------------------
    # Classification diagnosis
    # --------------------------------------------------------

    classification_result = results_df[
        results_df["task"] == "classification"
    ].sort_values(
        "macro_f1",
        ascending=False,
    )

    if classification_result.empty:
        raise ValueError(
            "No classification baseline result found."
        )

    best_classification = (
        classification_result.iloc[0]
    )

    best_classification_name = (
        best_classification["model"]
    )

    X_train, y_train = prepare_target_dataset(
        train_df,
        CLASSIFICATION_TARGET,
    )

    X_validation, y_validation = (
        prepare_target_dataset(
            validation_df,
            CLASSIFICATION_TARGET,
        )
    )

    # Make target encoding deterministic.
    classes = sorted(
        y_train.unique().tolist()
    )

    class_to_number = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    y_train_encoded = y_train.map(
        class_to_number
    )

    y_validation_encoded = (
        y_validation.map(class_to_number)
    )

    classification_models = (
        get_classification_models()
    )

    if best_classification_name not in (
        classification_models
    ):
        raise ValueError(
            "Unsupported classification model: "
            f"{best_classification_name}"
        )

    classification_model = (
        classification_models[
            best_classification_name
        ]
    )

    # Logistic / RF / XGB all have their own training
    # approach in the baseline script. For the diagnostic
    # refit, use the same model family.
    if best_classification_name == "XGBClassifier":

        class_counts = (
            y_train_encoded.value_counts()
        )

        total = len(y_train_encoded)
        number_of_classes = (
            len(class_counts)
        )

        class_weights = {
            class_name: total
            / (number_of_classes * count)
            for class_name, count
            in class_counts.items()
        }

        sample_weights = (
            y_train_encoded.map(
                class_weights
            ).to_numpy()
        )

        classification_model.fit(
            X_train,
            y_train_encoded,
            sample_weight=sample_weights,
        )

    else:

        classification_model.fit(
            X_train,
            y_train_encoded,
        )

    distribution, classification_diagnostic = (
        print_classification_diagnosis(
            y_validation=y_validation,
            model_name=best_classification_name,
            model=classification_model,
            X_train=X_train,
            y_train=y_train_encoded,
            X_validation=X_validation,
            y_validation_encoded=y_validation_encoded,
        )
    )

    diagnostics.append(
        classification_diagnostic
    )

    distribution["target"] = (
        CLASSIFICATION_TARGET
    )

    all_class_distributions.append(
        distribution
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    diagnostics_df.to_csv(
        DIAGNOSTICS_FILE,
        index=False,
    )

    if all_class_distributions:
        class_distribution_df = (
            pd.concat(
                all_class_distributions,
                ignore_index=True,
            )
        )

        class_distribution_df.to_csv(
            CLASS_DISTRIBUTION_FILE,
            index=False,
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC REPORTS SAVED")
    print("=" * 70)

    print(
        f"Diagnostics: "
        f"{DIAGNOSTICS_FILE}"
    )

    print(
        f"Class distribution: "
        f"{CLASS_DISTRIBUTION_FILE}"
    )


if __name__ == "__main__":
    main()