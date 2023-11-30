import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
#import lightgbm as lgb
import numpy as np
from math import log, e
import seaborn as sns
'''
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

def create_box_plot(df, x_value, y_value):
    # Plotting boxplot using seaborn
    sns.boxplot(data=df, x='city', y='age')
    plt.title('Boxplot of Age by City')
    plt.xlabel('City')
    plt.ylabel('Age')
    plt.show()

def get_descriptive_statistics(df):
    print(df.describe())
    print(df.dtypes)

def get_corr_matrix(df):
    df_corr = df.corr()
    fig, ax = plt.subplots(figsize=(14,8))
    ax = sns.heatmap(df_corr, annot = True)

def vizualize_distribution(df, column):
    plt.hist(df[column], bins=10)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def visualize_category_differences(df, col1, col2):
    plt.bar(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.xticks(rotation=45)
    plt.show()

def visualize_scatter_plot(df, col1, col2):
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()
'''

def remove_columns_with_na(df, threshold):
    missing_percentages = df.isnull().sum() / len(df) * 100

    # Get the column names that exceed the threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    df= df.drop(columns_to_drop, axis=1)
    return df
#TODO create a function to output all of the above in a report


def columns_with_na(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns

    categorical_na_columns = [col for col in categorical_columns if df[col].isna().any()]
    numerical_na_columns = [col for col in numerical_columns if df[col].isna().any()]

    return categorical_na_columns, numerical_na_columns

