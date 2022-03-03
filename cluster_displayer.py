import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns

def old_cluster_counting(df: DataFrame):
    clusters = set()
    for cluster_id in df.c:
        if cluster_id not in clusters:
            clusters.add(cluster_id)
    return len(clusters)

def cluster_counting(df: DataFrame):
    return len(set(df.c))


def show_cluster(file_name: str, img_title: str = "", 
    x_label: str = "", y_label: str = ""):
    '''Takes the name of a CSV file with points characterized\n
    by x,y,c (i.e. their coordinates and respective clusters).\n
    It displays all the points coloured according to their cluster.\n
    
    Optionally you can display a title and axis labels\n
    (which are empty by default)'''
    # After clustering
    plt.figure()
    df = pd.read_csv(file_name)

    # call this func here to count the occurrences of the clusters
    # in order for getting the correct number of clusters for scatterplot
    k = cluster_counting(df)

    # passing as arguments the number of occurrences of the clusters
    sns.scatterplot(x = df.x, y = df.y, 
                    hue = df.c, 
                    palette = sns.color_palette("hls", n_colors = k) )
                    # ACHTUNG ATTENZIONE XIAOXIN ATTENTION
                    # This stupid function doesn't work if it doesn't get
                    # the exact number of residual cluster centroids
                    # It can happen that one centr. collapses into another
                    # and so this stupid-ass func sees a wrong number of clusters
                    # This could also be due to an incorrect init. of centroids
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(img_title)
    plt.show()


def main():

    file_name = "./clusters_100.csv"
    show_cluster(file_name, img_title="100 punti lol")
    

if __name__ == "__main__":
    main()