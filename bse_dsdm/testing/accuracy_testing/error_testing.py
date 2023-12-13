
# Example Usage
# For Binary Classification
y_true_binary = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
y_pred_binary = [1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
evaluate_classification_metrics(y_true_binary, y_pred_binary)

# For Multiclass Classification
y_true_multiclass = [0, 1, 2, 1, 0, 2, 0, 1, 2, 0]
y_pred_multiclass = [0, 1, 2, 1, 0, 2, 0, 1, 2, 0]
evaluate_classification_metrics(y_true_multiclass, y_pred_multiclass)