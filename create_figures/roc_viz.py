import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def create_plot():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # convert output to binary
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # add noisy features to make the problem harder
    random_state = np.random.RandomState(1)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

    # learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel="linear", probability=True, random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plot the ROC curves
    plt.figure(figsize=(6,6))
    lw = 2
    plt.plot(0, 1, "o", color="grey", lw=lw, label="Optimal Point")
    plt.plot([0, 0, 1], [0, 1, 1], color="black", lw=lw, linestyle="dotted", label="ROC curve (AUC = 1.00)")
    plt.plot(fpr[2], tpr[2], color="black", lw=lw, label="ROC curve (AUC = %0.2f)" % roc_auc[2])
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--", label="ROC curve (AUC = 0.50)")

    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # show the legend
    plt.legend(loc="lower right")

    # show the plot
    plt.show()

    # save the plot
    # plt.savefig('graphs/roc_visualized.png', dpi=600)

if __name__ == '__main__':
    create_plot()
    print('done')