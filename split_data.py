import numpy as np
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=None):
    # Ensure that X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)
   
    # Check that the shapes of X and y are consistent
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y must be the same.")
   
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
   
    return X_train, X_test, y_train, y_test