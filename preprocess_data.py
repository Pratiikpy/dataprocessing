import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(X, y, missing_values='mean', binarize_threshold=0, scaling_method='standard', transform_method='yeo-johnson'):
    """
    Preprocesses the input data by imputing missing values, binarizing features, scaling numerical features, encoding categorical features, and rebalancing the data if necessary.
    
    Parameters:
    - X: a 2D numpy array representing the input data, with rows as samples and columns as features
    - y: a 1D numpy array representing the labels for the input data
    - missing_values: a string indicating the strategy to use for imputing missing values ('mean', 'median', 'most_frequent') or the value to impute ('constant')
    - binarize_threshold: a float indicating the threshold value for binarizing features (if 0, no binarization is performed)
   scaling_method: a string indicating the scaling method to use ('standard', 'minmax')
    - transform_method: a string indicating the transformation method to use ('yeo-johnson', 'box-cox')
    
    Returns:
    - X_preprocessed: a 2D numpy array representing the preprocessed input data, with rows as samples and columns as features
    - y_balanced: a 1D numpy array representing the rebalanced labels for the input data
    """
    # Impute missing values
    if missing_values in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=missing_values)
        X_imputed = imputer.fit_transform(X)
    else: # 'constant'
        imputer = SimpleImputer(strategy='constant', fill_value=missing_values)
        X_imputed = imputer.fit_transform(X)
        
    # Binarize features
    if binarize_threshold > 0:
        binarizer = Binarizer(threshold=binarize_threshold)
        X_binarized = binarizer.fit_transform(X_imputed)
    else:
        X_binarized = X_imputed
        
    # Separate input data into numerical and categorical features
    num_features = X.select_dtypes(include=np.number).columns
    cat_features = X.select_dtypes(exclude=np.number).columns
    
    # Scale numerical features
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    X_scaled_num = scaler.fit_transform(X_binarized[:, num_features])
    
    # Transform numerical features
    if transform_method == 'yeo-johnson':
        transformer = PowerTransformer(method='yeo-johnson')
    elif transform_method == 'box-cox':
        transformer = PowerTransformer(method='box-cox')
    X_transformed_num = transformer.fit_transform(X_scaled_num)
    
    # Encode categorical features using OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_scaled_cat = encoder.fit_transform(X_binarized[:, cat_features])
    
    # Concatenate the preprocessed numerical and categorical features
    X_scaled = np.concatenate((X_transformed_num, X_scaled_cat.toarray()), axis=1)
    
    # Check if the data is imbalanced
    if len(np.unique(y)) == 2: # binary classification
        class_counts = np.bincount(y)
        if np.min(class_counts) / np.max(class_counts) < 0.1: # if data is imbalanced
                        # Rebalance the data using oversampling or undersampling
            if class_counts[0] < class_counts[1]: # if class 0 is the minority class
                oversampler = RandomOverSampler(sampling_strategy=0.1)
                X_scaled, y_balanced = oversampler.fit_resample(X_scaled, y)
            else: # if class 1 is the minority class
                undersampler = RandomUnderSampler(sampling_strategy=0.1)
                X_scaled, y_balanced = undersampler.fit_resample(X_scaled, y)
        else: # if data is not imbalanced
            y_balanced = y
    else: # multi-class classification
        y_balanced = y
    
    return X_scaled, y_balanced