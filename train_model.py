import numpy as np
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

def train_model(X, y, missing_values='mean', binarize_threshold=0, scaling_method='standard', transform_method='none', model_type='linear', hyperparameters=None):
    # Preprocess the data
    X_preprocessed, y_balanced = preprocess_data(X, y, missing_values=missing_values, binarize_threshold=binarize_threshold, scaling_method=scaling_method, transform_method=transform_method)
   
    # Create the model
    if model_type == 'linear':
        model = LinearModel()
    elif model_type == 'tree':
        model = DecisionTreeModel()
    # ... add additional model types as needed
   
    # Train the model using hyperparameter tuning, if desired
    if hyperparameters is not None:
        # Define the search space
        param_grid = hyperparameters
       
        # Create the random search object
        random_search = RandomizedSearchCV(model, param_grid, cv=3)
       
        # Fit the random search object to the data
        random_search.fit(X_preprocessed, y_balanced)
       
        # Get the best estimator
        best_estimator = random_search.best_estimator_
       
        # Evaluate the model using cross-validation
        scores = cross_validate(best_estimator, X_preprocessed, y_balanced, cv=5,
                                scoring=('accuracy', 'precision', 'recall', 'f1', 'r2'))

      # Print the mean and standard deviation of the evaluation scores
        print(f'Accuracy: {np.mean(scores["test_accuracy"]):.3f} (+/- {np.std(scores["test_accuracy"]):.3f})')
        print(f'Precision: {np.mean(scores["test_precision"]):.3f} (+/- {np.std(scores["test_precision"]):.3f})')
        print(f'Recall: {np.mean(scores["test_recall"]):.3f} (+/- {np.std(scores["test_recall"]):.3f})')
        print(f'F1: {np.mean(scores["test_f1"]):.3f} (+/- {np.std(scores["test_f1"]):.3f})')
        print(f'R2: {np.mean(scores["test_r2"]):.3f} (+/- {np.std(scores["test_r2"]):.3f})')
       
        # Return the best estimator
        return best_estimator
    else:
        # Train the model without hyperparameter tuning
        model.fit(X_preprocessed, y_balanced)
       
        # Return the trained model
        return model