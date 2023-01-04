import matplotlib.pyplot as plt

def plot_feature_importances(model, feature_names, n_features=10):
    # Check that the model has a `feature_importances_` attribute
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have a feature_importances_ attribute.")
       
    # Check that the model is a tree-based model
    if not isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier)):
        raise TypeError("Model must be a tree-based model (i.e. DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier).")
       
    # Get the feature importances
    importances = model.feature_importances_
   
    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]
   
    # Get the top n_features indices
    top_n = indices[:n_features]
   
    # Create a barplot of the feature importances
    plt.figure(figsize=(10, 5))
    plt.bar(range(n_features), importances[top_n], align='center')
    plt.xticks(range(n_features), [feature_names[i] for i in top_n], rotation=90)
    plt.xlim([-1, n_features])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.show()
