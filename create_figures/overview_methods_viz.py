import matplotlib.pyplot as plt
import pandas as pd


def create_overview_methods():
    # read in the csv file with the data
    df = pd.read_csv('datasets/meta/overview_methods.csv')
    
    # create the barplot grouped by category
    plt.bar(df.loc[df['Category']=='Classical Statistical Models',"Method"],df.loc[df['Category']=='Classical Statistical Models',"Number"], 
            color='black', label='Classical Statistical Models')
    plt.bar(df.loc[df['Category']=='Machine Learning and Artificial Intelligence Models',"Method"],
            df.loc[df['Category']=='Machine Learning and Artificial Intelligence Models',"Number"], 
            color='gray', label='Machine Learning and Artificial Intelligence Models')
    plt.bar(df.loc[df['Category']=='Other Machine Learning Models',"Method"],df.loc[df['Category']=='Other Machine Learning Models',"Number"], 
            color='gray', label='Other Machine Learning Models')
    
    # create a legend
    plt.legend()
    
    # create labels for each bar
    label = 'n=' + df.loc[:,'Number'].astype(str)
    
    # create text below each bar with 90 degrees rotation
    plt.xticks([el for el in range(len(label))], df.loc[:,'Method'], rotation=90)
    
    # add the label text to each bar
    for i in range(len(label)):
        plt.text(x=df.loc[i,'Method'], y=df.loc[i,'Number'], s=label[i], size=10, ha='center', va='bottom')
    
    # adjust the margins
    plt.subplots_adjust(bottom= 0.35, top = 0.95, right = 0.7)
    
    # adjust figuresize
    fig = plt.gcf()
    fig.set_size_inches(16, 6)
    
    # add labels to the plot
    plt.ylabel('Number of applications in previous research')
    
    # save as png
    plt.savefig('graphs/overview_methods.png', dpi=600)
    
    # visualize the graphic
    # plt.show()
    
if __name__ == '__main__':
    create_overview_methods()
    print('done')
