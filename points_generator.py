import random
from re import X
import matplotlib.pyplot as plt
import numpy as np
import math

N = 1024 # number of points you desire to generate
N_DIM = 2
PI = 3.1415926535
TWO_PI = 6.283185307

def rand_num_gen(center: float = 0.0, tot_range: float = 1.0):
    '''Outputs a uniform random variable, \n
    centered by default in 0.0, in range -0.5 and 0.5'''
    return random.uniform(center - tot_range/2, center + tot_range/2)

def point_generator():
    '''Deprecated function used for generating points.txt'''
    with open("points.txt", "+a") as out_file:  
        # change to "w" here ^^ if you wanna override all older points
        for point in range(N):
            for dim in range(N_DIM):
                if dim == 0:
                    x = rand_num_gen(5.0, 3.0)      # x center
                if dim == 1:
                    x = rand_num_gen(3.0, 3.0)     # y center
                out_file.write(f"{x} ")
            out_file.write("\n")

def point_generator_circ(n_point_per_clust: int, k: int, file_name: str, cluster_centers,
        n_dim : int = 2, rho_max: float = 8.0, cl_cntr_max: float = 30.0):
    '''
    Generates k clusters with n points per cluster in file_name \n
    (you can choose the max radius from center for each cluster) \n
    The cluster center is generated in interval [-10, 10] for both \n
    x and y coordinates...
    '''

    with open(file_name, "w") as out_file:  
        # change to "w" here ^^ if you wanna override all older points
        # or "+a" if you wanna add to older points
        for cluster in range(k):
            # generate new center for each cluster
            # cluster_center = [random.uniform(-cl_cntr_max, cl_cntr_max), 
            #     random.uniform(-cl_cntr_max, cl_cntr_max)]
            cluster_center = cluster_centers[cluster]
            for point in range(n_point_per_clust):
                rho = random.uniform(0.0, rho_max)
                theta = random.uniform(0.0, TWO_PI)
                x = math.cos(theta) * rho  + cluster_center[0]
                y = math.sin(theta) * rho  + cluster_center[1]
                out_file.write(f"{x} {y}\n")

def cluster_saver(file_name: str, cluster_centers, k: int):
    # reads from file and takes values into lists
    with open(file_name, "w") as out_file:
        for cluster in range(k):
            x, y = cluster_centers[cluster]
            out_file.write(f"{x} {y} \n")




def point_plotter(file_name: str):
    # file_name = input("Which file you want to read points from: ") #if you want user input
    # file_name = "points2.txt"

    # reads from file and takes values into lists
    with open(file_name, "r") as in_file:
        x, y = [], []
        for line in in_file:
            line_splitted = line.split()
            x.append(float(line_splitted[0]))
            y.append(float(line_splitted[1]))

    # fig, ax = plt.subplot()
    # ax.plot(x, y)
    # fig.show()
    plt.scatter(x, y)
    plt.axis()
    plt.show()

def count_points(file_name: str):
    # file_name = "points2.txt"
    #reads how many points there are (num of lines)
    with open(file_name, "r") as in_file:
        i = 0
        for line in in_file:
            i += 1
        print(f"There are {i} points in {file_name}")




def main():

    file_name = "points_1024.txt"
    N = 128
    K = 8
    cluster_centers = [[-150, -150], [20, 25], [-8, -8], [-60,-75], [-30,41], [24,5], [-49,-6], [99,99]]
    cluster_saver("init_centroids_8.txt", cluster_centers, K)
    point_generator_circ(N, K, file_name, cluster_centers)

    
    # un buon rapporto tra raggio e cl_cntr_max è di 1.35/5.0 (quando hai meno di 10 cluster)
    # point_generator_circ(10, 10, file_name, rho_max = 1.35, cl_cntr_max= 5.0)  
    count_points(file_name)
    point_plotter(file_name)
    


if __name__ == "__main__":
    main()





