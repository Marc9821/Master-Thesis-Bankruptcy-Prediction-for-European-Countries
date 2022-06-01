import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
from mlxtend.plotting import plot_decision_regions


def knn(data):
    # reading in the data, setting x and y
    x = data[['X','Y']].values
    y = data['class'].astype(int).values
    
    # setting k to 1 and 20
    i = 1
    ks  = [1, 20]
    for k in ks:
        #specify subplots and adding axes annotations
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.subplot(1, len(ks), i)
        
        # running the KNN model
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf.fit(x, y)
        
        # plot decision region
        plot_decision_regions(x, y, clf=clf, legend=2, colors='black,gray')

        # setting plot title
        plt.title('KNN with K='+ str(k))
        i += 1
    
    # adjust figuresize
    fig = plt.gcf()
    fig.set_size_inches(len(ks)*6, 6)
    
    # show images    
    # plt.show()
    
    # save as png
    plt.savefig('graphs/knn_visualized.png', dpi=600)

if __name__ == '__main__':
    # reading in the data from csv file
    data = pd.read_csv('datasets/knn/ushape.csv')
    knn(data)