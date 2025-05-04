import helper
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from sklearn.metrics import RocCurveDisplay
import pandas as pd


def aggregate_predict_by_score(
        models_predictions: List[np.ndarray],
        models_metrics: List[List[float]],
        models_weights: np.ndarray,
        precision_weight: float = 0.6,
        f1_weight: float = 0.4
) -> np.ndarray:

    """
    Calculate aggregated predictions using model weights and performance metrics.

    Each prediction is adjusted according to model importance (weights) and
    performance (weighted metrics).

    Parameters
    ----------
    models_predictions : List[np.ndarray]
        A list containing prediction arrays for each model.
    models_metrics : List[List[float]]
        A list containing [Precision, F1] metrics for each model.
    models_weights : np.ndarray
        Array containing the weight of each model (should sum to 1).
    precision_weight : float, optional
        Weight assigned to Precision metric, by default 0.6
    f1_weight : float, optional
        Weight assigned to F1 metric, by default 0.4

    Returns
    -------
    np.ndarray
        Binary array of final predictions after model integration.

    Notes
    -----
    The function uses a threshold of 0.5 to convert probability scores to binary predictions.
    """

    if precision_weight + f1_weight != 1.0:
        print("Warning: Precision and F1 weights don't sum to 1.0!")

    # Calculate combined performance score for each model
    weighted_scores = np.array([
        metrics[0] * precision_weight + metrics[1] * f1_weight
        for metrics in models_metrics
    ])

    # Calculate weighted predictions
    weighted_predictions = [
        pred * models_weights[i] * weighted_scores[i]
        for i, pred in enumerate(models_predictions)
    ]

    # Combine predictions
    combined_weight_factor = np.sum(models_weights * weighted_scores)
    if combined_weight_factor == 0:
        raise ValueError("Combined weight factor is zero, cannot normalize predictions!")

    results = sum(weighted_predictions) / combined_weight_factor

    # Apply threshold for binary classification
    threshold = 0.5
    final_prediction = np.where(results >= threshold, 1, 0)

    return final_prediction


def aggregate_predict_by_vote(models_predictions: List[np.ndarray]) -> np.ndarray:

    """
    Generate final predictions using a voting mechanism.

    The voting logic works as follows:
    1. Sensor models (all but last model) vote - if 2/3 vote 1, the result is 1
    2. If sensor vote matches network model (last model), use that prediction
    3. Otherwise, predict 1 (potential attack)

    Parameters
    ----------
    models_predictions : List[np.ndarray]
        A list of prediction arrays from different models.
        The last array is assumed to be from the network traffic model.

    Returns
    -------
    np.ndarray
        Final binary predictions after voting integration.
    """

    if len(models_predictions) < 2:
        raise ValueError("Need at least two models for voting aggregation!")

    # Calculate majority vote from sensor models (all but last model)
    num_samples = len(models_predictions[0])
    majority_votes = np.zeros(num_samples, dtype=int)

    # Count votes for each sample
    for i in range(num_samples):
        # Sum votes from sensor models (excluding the last model)
        votes_sum = sum(model[i] for model in models_predictions[:-1])

        # If 2/3 or more voted 1, majority is 1
        required_votes = 2  # Assuming 3 sensor models
        majority_votes[i] = 1 if votes_sum >= required_votes else 0

    # Compare majority vote with network model prediction
    network_predictions = models_predictions[-1]
    final_prediction = np.where(
        majority_votes == network_predictions,
        majority_votes,  # When both agree
        np.ones_like(majority_votes)  # When they disagree, predict 1 (potential attack)
    )

    return final_prediction


def get_predictions_and_metrics(
        local_models: List[str],
        sensor_test: pd.DataFrame,
        global_model: str,
        network_test: pd.DataFrame,
        roc: bool = True
) -> Tuple[List[np.ndarray], List[List[float]]]:

    """
    Test models and get their predictions and performance metrics.

    Parameters
    ----------
    local_models : List[str]
        Paths to local model files.
    sensor_test : pd.DataFrame
        Test data for sensor models with target in the last column.
    global_model : str
        Path to global model file.
    network_test : pd.DataFrame
        Test data for network model with target in the last column.
    roc : bool, optional
        Whether to plot ROC curves, by default True

    Returns
    -------
    Tuple[List[np.ndarray], List[List[float]]]
        A tuple containing:
        - List of model predictions
        - List of model metrics [precision, f1] for each model
    """

    # Extract features and target variables
    sensor_x = sensor_test.iloc[:, :-1]
    sensor_y = sensor_test.iloc[:, -1]
    network_x = network_test.iloc[:, :-1]
    network_y = network_test.iloc[:, -1]

    models_metrics = []
    models_predictions = []

    # Set up ROC curve plot if requested
    if roc:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Test local models (sensor models)
    for i, model_path in enumerate(local_models):
        try:
            local_model = joblib.load(model_path)
            s_predict = local_model.predict(sensor_x)
            models_predictions.append(s_predict)

            print(f"\nClient {i + 1} Prediction Results:")
            accuracy, precision, recall, f1 = helper.get_metrics(sensor_y, s_predict, printout=True)
            models_metrics.append([precision, f1])

            # Add ROC curve if requested
            if roc and hasattr(local_model, 'predict_proba'):
                s_predict_proba = local_model.predict_proba(sensor_x)[:, 1]
                RocCurveDisplay.from_predictions(
                    sensor_y,
                    s_predict_proba,
                    name=f'Client {i + 1}',
                    ax=ax
                )
        except Exception as e:
            print(f"Error processing local model {i + 1}: {e}")
            raise

    # Test global model (network model)
    try:
        network_model = joblib.load(global_model)
        n_predict = network_model.predict(network_x)
        models_predictions.append(n_predict)

        print(f"\nServer Prediction Results:")
        accuracy, precision, recall, f1 = helper.get_metrics(network_y, n_predict, printout=True)
        models_metrics.append([precision, f1])

        # Add ROC curve if requested
        if roc and hasattr(network_model, 'predict_proba'):
            n_predict_proba = network_model.predict_proba(network_x)[:, 1]
            RocCurveDisplay.from_predictions(
                network_y,
                n_predict_proba,
                name='Server',
                ax=ax
            )
    except Exception as e:
        print(f"Error processing network model: {e}!")
        raise

    # Finalize and display ROC plot if requested
    if roc:
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return models_predictions, models_metrics
