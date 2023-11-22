from sklearn.metrics import r2_score


r_squared = r2_score(y, predictions)
adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - k - 1)