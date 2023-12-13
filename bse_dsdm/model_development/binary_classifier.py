
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb
import numpy as np
from math import log, e
import seaborn as sns

def gini_impurity(value_counts):
    """
    Gini impurity measures the probability of incorrectly classifying a randomly chosen element in a dataset.
    * Scale: Gini impurity values range between 0 and 1, where 0 represents a perfectly pure dataset (all instances belong to the same class), and 0.5 represents maximum impurity (an equal distribution of instances across all classes).
    * Sensitivity to Imbalance: Gini impurity is sensitive to class imbalance. It tends to favor splits that create more balanced subsets.
    * Binary Decision: In each binary split, the Gini impurity is calculated for both resulting subsets, and the weighted sum of the impurities is used to evaluate the quality of the split.
    * Decision Tree Usage: Gini impurity is commonly used in decision tree algorithms such as CART (Classification and Regression Trees)
    """
    n = value_counts.sum()
    p_sum = 0
    for key in value_counts.keys():
        p_sum = p_sum  +  (value_counts[key] / n ) * (value_counts[key] / n )
    gini = 1 - p_sum
    return gini

def gini_split_a(attribute_name, class_name, df):
    attribute_values = df[attribute_name].value_counts()
    gini_A = 0
    for key in attribute_values.keys():
        df_k = df[class_name][df[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = df.shape[0]
        gini_A = gini_A + (( n_k / n) * gini_impurity(df_k))
    return gini_A

def get_entropy(column_of_labels):
    """
    Entropy is a measure of the disorder or randomness in a dataset. In decision trees, entropy quantifies the uncertainty about the class labels of the target variable.

    * Scale: Entropy values range between 0 and 1, where 0 represents perfect order (all instances belong to a single class), and 1 represents maximum disorder (an equal distribution of instances across all classes).
    * Sensitivity to Imbalance: Like Gini impurity, entropy is also sensitive to class imbalance and may lead to more balanced splits.
    * Information Theory: Entropy is rooted in information theory and reflects the average number of bits needed to represent the class labels of instances in the dataset.
    * Decision Tree Usage: Entropy is commonly used in decision tree algorithms such as ID3 (Iterative Dichotomiser 3) and C4.5.
    """
    n_labels = len(column_of_labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(column_of_labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

def fit_decision_tree(X,y):
    # Train a Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=2, max_depth = 3) # Restric to 3 levels
    clf.fit(X, y)

def fit_random_forest(X,y):
    # Initialize a random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=1)

    # Train the random forest model
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

def fit_light_gbm(X,y):
    # Initialize a LightGBM classifier
    lgb_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=1)

    # Train the LightGBM model
    lgb_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lgb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

def plot_decision_tree(clf, feature_names, class_names):
    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
    plt.show()

def plot_confusion_matrix():
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

