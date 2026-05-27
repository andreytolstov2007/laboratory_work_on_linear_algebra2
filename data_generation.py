import random
import math

def Generate_synthetic_data(n_samples=500, random_state=42):
    random.seed(random_state)
    mean1 = [-1.5, -1.5]
    mean2 = [1.5, 1.5]
    cov = [[0.8, 0.2], [0.2, 0.8]]
    X = []
    y = []
    n_class0 = n_samples // 2
    for _ in range(n_class0):
        x1 = random.gauss(mean1[0], math.sqrt(cov[0][0]))
        x2 = random.gauss(mean1[1], math.sqrt(cov[1][1]))
        X.append([x1, x2])
        y.append(0)
   
    n_class1 = n_samples - n_class0
    for _ in range(n_class1):
        x1 = random.gauss(mean2[0], math.sqrt(cov[0][0]))
        x2 = random.gauss(mean2[1], math.sqrt(cov[1][1]))
        X.append([x1, x2])
        y.append(1)
   
    combined = []
    for i in range(len(X)):
        combined.append([X[i], y[i]])
   
    random.shuffle(combined)
    X_shuffled = []
    y_shuffled = []
    for i in range(len(combined)):
        X_shuffled.append(combined[i][0])
        y_shuffled.append(combined[i][1])
   
    return X_shuffled, y_shuffled

def Manual_train_test_split(X, y, test_size=0.3, random_state=42):
    random.seed(random_state)
    class0_indices = []
    class1_indices = []
    for i in range(len(y)):
        if y[i] == 0:
            class0_indices.append(i)
        else:
            class1_indices.append(i)
   
    n_test_class0 = int(len(class0_indices) * test_size)
    n_test_class1 = int(len(class1_indices) * test_size)
    test_idx0 = random.sample(class0_indices, n_test_class0)
    test_idx1 = random.sample(class1_indices, n_test_class1)
    all_test_indices = []
    for i in range(len(test_idx0)):
        all_test_indices.append(test_idx0[i])
    for i in range(len(test_idx1)):
        all_test_indices.append(test_idx1[i])
   
    test_indices_set = set(all_test_indices)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X)):
        if i in test_indices_set:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
   
    return X_train, X_test, y_train, y_test

def Manual_standardize(X_train, X_test):
    n_features = len(X_train[0])
    mean = []
    std = []
    for j in range(n_features):
        col = []
        for i in range(len(X_train)):
            col.append(X_train[i][j])
       
        col_sum = 0.0
        for value in col:
            col_sum = col_sum + value
        col_mean = col_sum / len(col)
        mean.append(col_mean)
       
        squared_diff_sum = 0.0
        for value in col:
            diff = value - col_mean
            squared_diff_sum = squared_diff_sum + (diff * diff)
        variance = squared_diff_sum / len(col)
        col_std = math.sqrt(variance)
       
        if col_std > 0:
            std.append(col_std)
        else:
            std.append(1.0)
   
    X_train_norm = []
    for i in range(len(X_train)):
        normalized_row = []
        for j in range(n_features):
            normalized_value = (X_train[i][j] - mean[j]) / std[j]
            normalized_row.append(normalized_value)
        X_train_norm.append(normalized_row)
   
    X_test_norm = []
    for i in range(len(X_test)):
        normalized_row = []
        for j in range(n_features):
            normalized_value = (X_test[i][j] - mean[j]) / std[j]
            normalized_row.append(normalized_value)
        X_test_norm.append(normalized_row)
   
    return X_train_norm, X_test_norm