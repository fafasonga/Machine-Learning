import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt


class ROC:
    """
    ROC curve builder class.
    Classes are assumed to be binary

    """
    # results is an numpy array formed by stacking together fpr, tpr and corresponding thresholds.
    # use results for analysis
    results = None

    def __init__(self, proba, true_labels, pos_label_value, pos_label=1):
        """
        Use these values in calc_tpr_fpr() method

        :param proba: numpy array of class probabilities
        :param true_labels: numpy array of true labels
        :param pos_label_value: The value of the positive label (usually 1)
        :param pos_label: The relative order of positive label in proba
        """
        self.proba = proba
        self.true_labels = true_labels
        self.pos_label_value = pos_label_value
        self.pos_label = pos_label
        print("pos_label", proba)


    def plot(self):
        """
        Plots an ROC curve using True Positive Rate and False Positive rate lists calculated from __calc_tpr_fpr
        Calculates and outputs AUC score on the same graph
        """
        tpr, fpr, thresholds = self.__calc_tpr_fpr()
        self.results = np.column_stack((tpr, fpr, thresholds))

        # %%% TODO START YOUR CODE HERE %%%

        fig = plt.figure()
        plt.plot(fpr, tpr)
        fig.suptitle('ROC Plot')
        plt.xlabel('True Negative Rate')
        plt.ylabel('True Positive Rate')

        # %%% END YOUR CODE HERE %%%

    def __calc_tpr_fpr(self):
        """
        Calculates True Positive Rate, False Positive Rate and thresholds lists

        First, sorts probabilities of positive label in decreasing order
        Next, moving towards the least probability locates a threshold between instances with opposite classes
        (keeping instances with the same confidence value on the same side of threshold),
        computes TPR, FPR for instances above threshold and puts them in the lists

        :return:
        tpr: list
        fpr: list
        thresholds: list
        """
        # %%% TODO START YOUR CODE HERE %%%

        tpr = [0]
        fpr = [0]
        thresholds = [1]

        TP = 0
        FN = (self.true_labels == 1).sum()
        TN = (self.true_labels == 0).sum()
        FP = 0

        prob_of_positive = self.proba[:, self.pos_label]
        index_sorted = np.argsort(-prob_of_positive)

        sorted_prob = prob_of_positive[index_sorted]
        sorted_labels = self.true_labels[index_sorted]

        for i in range(0, len(sorted_labels) - 1):
            if sorted_labels[i] == 1:
                FN -= 1
                TP += 1
            if sorted_labels[i] == 0:
                TN -= 1
                FP += 1

            if sorted_labels[i] != sorted_labels[i+1]:
                tpr.append(TP/(TP + FN))
                fpr.append((FP/(TN + FP)))
                thresholds.append(sorted_prob[i])

        thresholds.append(0)
        tpr.append(1)
        fpr.append(1)

        # print("FPR = ", fpr)
        # print("TPR = ", tpr)

        return tpr, fpr, thresholds


        # %%% END YOUR CODE HERE %%%


def stratified_train_test_split(X, Y, test_size, random_seed=None):
    """
    Performs the stratified train/test split
    (with the same (!) inter-class ratio in train and test sets as compared to original set)
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        test_size: number between 0 and 1, specifies the relative size of the test_set
        random_seed: random_seed

    returns:
        X_train
        X_test
        Y_train
        Y_test
    """
    if test_size < 0 or test_size > 1:
        raise Exception("Fraction for split is not valid")

    np.random.seed(random_seed)

    # %%% TODO START YOUR CODE HERE %%%

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    zeros = np.sum(Y == 0)
    print(zeros)
    num_zeros = int(zeros * test_size)
    ones = np.sum(Y == 1)
    num_ones = int(ones * test_size)

    # For X Test

    index_zero = np.random.choice(range(zeros), size=num_zeros,replace=False)
    choice_x_zero = X[Y == 0]
    test_x_zero = choice_x_zero[index_zero]

    index_one = np.random.choice(range(ones), size=num_ones, replace=False)
    choice_x_one = X[Y == 1]
    test_x_one = choice_x_one[index_one]

    X_test = np.concatenate([test_x_one, test_x_zero])


    # For Y Test

    choice_y_zero = Y[Y == 0]
    test_y_zero = choice_y_zero[index_zero]

    choice_y_one = Y[Y == 1]
    test_y_one = choice_y_one[index_one]

    y_test = np.concatenate([test_y_zero, test_y_one])


    # For X Train

    train_index_zero = [i for i in range(num_zeros) if i not in index_zero]
    choose_x_zero = X[Y == 0]
    train_x_zero = choose_x_zero[train_index_zero]

    train_index_ones = [i for i in range(num_ones) if i not in index_one]
    choose_x_one = X[Y == 1]
    train_x_one = choose_x_one[train_index_ones]

    X_train = np.concatenate([train_x_zero, train_x_one])

    # For Y Train

    choose_y_zero = Y[Y == 0]
    train_y_zero = choose_y_zero[train_index_zero]

    choose_y_one = Y[Y == 1]
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
    # %%% END YOUR CODE HERE %%%


data = load_breast_cancer()

# Pre-processing: Exchange labels - make malignant 1, benign 0
data['target'] = np.array(data['target'], dtype=int) ^ 1

X_train, X_test, y_train, y_test = stratified_train_test_split(data['data'], data['target'], 0.3, 10)

# Check that the ratio is preserved
print("Inter-class ratio in original set:", len(np.argwhere(data['target'] == 1))/len(np.argwhere(data['target'] == 0)))
print("Inter-class ratio in train set:", len(np.argwhere(y_train == 1))/len(np.argwhere(y_train == 0)))
print("Inter-class ratio in test set:", len(np.argwhere(y_test == 1))/len(np.argwhere(y_test == 0)))
print('\n')

# We pick Logistic Regression because it outputs probabilities
# Try different number of iterations to change ROC curve
model = LogisticRegression(max_iter=5)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
y_pred = model.predict(X_test)
print("Classifier's Accuracy:", accuracy_score(y_test, y_pred))

# Build an ROC curve
roc = ROC(probabilities, y_test, 1)
roc.plot()
# Explore the results
results = roc.results

# Use scikitplot library to compare ROC curve with the one you are getting
skplt.metrics.plot_roc_curve(y_test, probabilities)
plt.show()



