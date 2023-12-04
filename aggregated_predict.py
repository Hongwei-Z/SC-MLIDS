import helper
import joblib
import numpy as np


# Testing models using testing sets to obtain predictions and performance metrics
def get_predictions_and_metrics(local_models, sensor_test, global_model, network_test):
    sensor_X = sensor_test.iloc[:, :-1]
    sensor_y = sensor_test.iloc[:, -1]
    network_X = network_test.iloc[:, :-1]
    network_y = network_test.iloc[:, -1]

    # Test the local models and show the metrics
    models_metrics = []
    models_predictions = []

    for i in range(len(local_models)):
        local_model = joblib.load(local_models[i])
        predict = local_model.predict(sensor_X)
        models_predictions.append(predict)
        print(f"Local Model {i + 1} Prediction Results:")
        accuracy, precision, recall, f1 = helper.get_metrics(sensor_y, predict, printout=True)
        models_metrics.append([precision, f1])

    # Test the global model and show the metrics
    network_model = joblib.load(global_model)
    predict = network_model.predict(network_X)
    models_predictions.append(predict)
    print(f"Global Model Prediction Results:")
    accuracy, precision, recall, f1 = helper.get_metrics(network_y, predict, printout=True)
    models_metrics.append([precision, f1])

    return models_predictions, models_metrics


'''
Calculating the aggregated predictions using the weights of the models 
and their performance metrics, which means that each prediction is adjusted 
according to the importance (weights) and performance (weighted metrics) of the model.
'''
def aggregate_predict_by_score(models_predictions, models_metrics, models_weights, precision_weight=0.6, f1_weight=0.4):
    # Calculate weighted scores using precision and f1
    weighted_scores = []
    for m in models_metrics:
        score = m[0] * precision_weight + m[1] * f1_weight
        weighted_scores.append(score)

    # The prediction results of each model are multiplied by its weights and weighted scores
    weighted_scores = np.array(weighted_scores)
    weighted_predictions = []
    for i in range(len(models_predictions)):
        weighted_pred = models_predictions[i] * models_weights[i] * weighted_scores[i]
        weighted_predictions.append(weighted_pred)

    '''
    The sum of the weighted predictions for all models is calculated and divided by 
    the sum of all model weights and weighted scores. This gives an average value that 
    reflects the combined predictions of all models.
    '''
    results = sum(weighted_predictions) / np.sum(models_weights * weighted_scores)

    # Converting predictions to binary classification results
    threshold = 0.5
    final_prediction = np.where(results >= threshold, 1, 0)

    return final_prediction


'''
Voting based on model predictions, if the majority of predictions in the local model are 1, 
then the local result is 1. If the local result is the same as the global model prediction, 
then it is the final prediction, otherwise, the global model prevails.
'''
def aggregate_predict_by_vote(models_predictions):
    majority_votes = []

    # Get the local vote results
    for i in range(len(models_predictions[0])):
        result_sum = models_predictions[0][i] + models_predictions[1][i] + models_predictions[2][i]
        if result_sum >= 2:
            vote = 1
        else:
            vote = 0
        majority_votes.append(vote)
    majority_votes = np.array(majority_votes)

    # If the majority vote result is the same as the global model's prediction, the majority vote result is used;
    # otherwise, the global model's prediction is used
    final_prediction = []
    for j in range(len(majority_votes)):
        if majority_votes[j] == models_predictions[-1][j]:
            final_prediction.append(majority_votes[j])
        else:
            final_prediction.append(models_predictions[-1][j])
    final_prediction = np.array(final_prediction)

    return final_prediction
