from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt


def create_plot():
    # generate 2 class dataset
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)

    # split into train/test sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=1)

    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(train_X, train_y)

    # predict probabilities
    lr_probs = model.predict_proba(test_X)

    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]

    # predict class values
    yhat = model.predict(test_X)
    lr_precision, lr_recall, _ = precision_recall_curve(test_y, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    # plot the precision-recall curves
    plt.figure(figsize=(6,6))
    lw = 2
    plt.plot(1, 1, "o", color="grey", lw=lw, label="Optimal Point")
    plt.plot([0,1,1], [1,1,0.5], color="black", lw=lw, linestyle="dotted", label="PR curve (AUC = 1.00)")
    plt.plot(lr_recall, lr_precision, color="black", lw=lw, label='PR curve (AUC = %0.2f)' % lr_auc)
    plt.plot([0, 1], [0.5, 0.5], color="black", lw=lw, linestyle='--', label="PR curve (AUC = 0.50)")

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # show the legend
    plt.legend()

    # show the plot
    plt.show()

    # save the plot
    # plt.savefig('graphs/precision_recall_visualized.png', dpi=600)
    
if __name__ == '__main__':
    create_plot()
    print('done')