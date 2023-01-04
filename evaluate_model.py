import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y, metric_names=['accuracy', 'precision', 'recall', 'f1']):
    """Calculate evaluation metrics for the given model.
   
    Parameters
    ----------
    model: object
        A trained model object.
    X: array-like, shape (n_samples, n_features)
        The input samples.
    y: array-like, shape (n_samples,)
        The target values.
    metric_names: list, optional
        A list of names of evaluation metrics to calculate. Available options are:
        ['accuracy', 'precision', 'recall', 'f1'].
   
    Returns
    -------
    scores: dict
        A dictionary of evaluation scores, with the metric names as keys and the
        corresponding scores as values.
    """
    # Initialize a dictionary to store the evaluation scores
    scores = {}
   
    # Make predictions on the input data
    y_pred = model.predict(X)
   
    # Calculate the evaluation scores
    if 'accuracy' in metric_names:
        scores['accuracy'] = accuracy_score(y, y_pred)
    if 'precision' in metric_names:
        scores['precision'] = precision_score(y, y_pred)
    if 'recall' in metric_names:
        scores['recall'] = recall_score(y, y_pred)
    if 'f1' in metric_names:
        scores['f1'] = f1_score(y, y_pred)
   
    # Return the dictionary of evaluation scores
    return scores