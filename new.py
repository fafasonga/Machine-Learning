import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import numpy as np
from pandas import read_csv

path = './crime.csv'
dataset = read_csv(path, sep=';', na_values=".", header='infer', dtype=str)
# print (dataset)


def plot(X, proba):
        """
        Plots an ROC curve using True Positive Rate and False Positive rate lists calculated from __calc_tpr_fpr
        Calculates and outputs AUC score on the same graph
        """
        tpr, fpr, thresholds = calc_tpr_fpr(X, proba)
        results = np.column_stack((tpr, fpr, thresholds))

        fig = plt.figure()
        plt.plot(fpr, tpr)
        fig.suptitle('ROC Plot')
        plt.xlabel('True Negative Rate')
        plt.ylabel('True Positive Rate')


def calc_tpr_fpr(element, proba):
    tpr = [0]
    fpr = [0]
    thresholds = [1]

    TP = 0
    FN = (element == 'YES').sum()
    TN = (element == 'NO').sum()
    FP = 0

    pos_label = [i for i in element if i == 'YES']


    prob_of_positive = proba[:, pos_label]
    print(prob_of_positive)
    index_sorted = np.argsort(-prob_of_positive)

    sorted_prob = prob_of_positive[index_sorted]
    sorted_labels = element[index_sorted]
    print(sorted_labels)

    for i in range(0, len(sorted_labels) - 1):
        if sorted_labels[i] == 1:
            FN -= 1
            TP += 1
            if sorted_labels[i] == 0:
                TN -= 1
                FP += 1

            if sorted_labels[i] != sorted_labels[i + 1]:
                tpr.append(TP / (TP + FN))
                fpr.append((FP / (TN + FP)))
                thresholds.append(sorted_prob[i])

        thresholds.append(0)
        tpr.append(1)
        fpr.append(1)

        print("FPR = ", fpr)
        print("TPR = ", tpr)

        return tpr, fpr, thresholds


def stratified_train_test_split(X_axis, Y_axis, test_size, random_seed=None):

    if test_size < 0 or test_size > 1:
        raise Exception("Fraction for split is not valid")

    np.random.seed(random_seed)

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    zeros = np.sum(X_axis == 'NO')
    num_zeros = int(zeros * test_size)
    ones = np.sum(X_axis == 'YES')
    num_ones = int(ones * test_size)

    # For X Test

    index_zero = np.random.choice(range(zeros), size=num_zeros,replace=False)
    choice_x_zero = Y_axis[X_axis == 'NO']
    test_x_zero = choice_x_zero[index_zero]

    index_one = np.random.choice(range(ones), size=num_ones, replace=False)
    choice_x_one = Y_axis[X_axis == 'YES']
    test_x_one = choice_x_one[index_one]

    X_test = np.concatenate([test_x_one, test_x_zero])

    # For Y Test

    choice_y_zero = X_axis[X_axis == 'NO']
    test_y_zero = choice_y_zero[index_zero]

    choice_y_one = X_axis[X_axis == 'YES']
    test_y_one = choice_y_one[index_one]

    y_test = np.concatenate([test_y_zero, test_y_one])

    # For X Train

    train_index_zero = [i for i in range(num_zeros) if i not in index_zero]
    choose_x_zero = Y_axis[X_axis == 'NO']
    train_x_zero = choose_x_zero[train_index_zero]

    train_index_ones = [i for i in range(num_ones) if i not in index_one]
    choose_x_one = Y_axis[X_axis == 'YES']
    train_x_one = choose_x_one[train_index_ones]

    X_train = np.concatenate([train_x_zero, train_x_one])

    # For Y Train

    choose_y_zero = X_axis[X_axis == 'NO']
    train_y_zero = choose_y_zero[train_index_zero]

    choose_y_one = X_axis[X_axis == 'YES']
    train_y_one = choose_y_one[train_index_ones]

    y_train = np.concatenate([train_y_one, train_y_zero])

    indeces = np.arange(len(y_train))
    np.random.shuffle(indeces)
    X_train = X_train[indeces]
    y_train = y_train[indeces]


    indeces = np.arange(len(y_test))
    np.random.shuffle(indeces)
    X_test = X_test[indeces]
    y_test = y_test[indeces]

    return X_train, X_test, y_train, y_test

X = np.array(dataset['PRIORITY'])
Y = np.array(dataset['TYPE'])
Z = np.array(dataset['NEIGHBOURHOOD'])
X_train, X_test, y_train, y_test = stratified_train_test_split(X, Y, 0.3, 10)

# Check that the ratio is preserved
print("Inter-class ratio in original set:", len(np.argwhere(X == 'YES'))/len(np.argwhere(X == 'NO')))
print("Inter-class ratio in train set:", len(np.argwhere(y_train == 'YES'))/len(np.argwhere(y_train == 'NO')))
print("Inter-class ratio in test set:", len(np.argwhere(y_test == 'YES'))/len(np.argwhere(y_test == 'NO')))
print('\n')

# We pick Logistic Regression because it outputs probabilities
# Try different number of iterations to change ROC curve
# model = LogisticRegression(max_iter=5)
# model.fit(X_train, y_train)
# probabilities = model.predict_proba(X_test)
# y_pred = model.predict(X_test)
# print(y_pred)
# print("Classifier's Accuracy:", accuracy_score(y_test, y_pred))
#
# # Build an ROC curve
# roc = ROC(probabilities, y_test, 1)
# roc.plot(fpr, tpr)
# # Explore the results
# results = roc.results
#
#
# # Use scikitplot library to compare ROC curve with the one you are getting
# skplt.metrics.plot_roc_curve(y_test, probabilities)
# plt.show()



