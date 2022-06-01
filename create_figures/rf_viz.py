import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pydotplus import graph_from_dot_data


# create classification report
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

def visualizeRf(max_depth):
    # fix dot.exe in PATH not working, set 'C:/Program Files/Graphviz/bin' to path of your graphviz installation
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    
    # read in dataset
    df = pd.read_csv('datasets/rf/data.csv')

    # remove 96% of data randomly from non-bankrupt sample, this is just to create balance of the two classes, note that this is just used for illustrative purposes
    df = df.drop(df[df['Bankrupt?']==0].sample(frac=0.96).index)

    # define X and y
    X = df.drop(['Bankrupt?'], axis=1)
    y = df['Bankrupt?']
    class_name = ['non-bankrupt','bankrupt']

    # training and testing set split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # initiate DecisionTreeClassifier with gini index criterion, max_depth is a tuning parameter
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=0, min_samples_split=40)
    clf_gini = clf_gini.fit(X_train, y_train)

    # print the scores
    print_score(clf_gini, X_train, y_train, X_test, y_test, train=True)
    print_score(clf_gini, X_train, y_train, X_test, y_test, train=False)

    # create DOT data
    dot_data = tree.export_graphviz(clf_gini, out_file=None, 
                                    feature_names=X.columns,  
                                    class_names=class_name,
                                    proportion=True,
                                    max_depth=max_depth,
                                    filled=False)

    # Draw graph and set size
    graph = graph_from_dot_data(dot_data)
    graph.set_size('"100!"')
    graph.write_png('graphs/tree_visualization.png')

if __name__ == "__main__":
    visualizeRf(3)
    print('done')
