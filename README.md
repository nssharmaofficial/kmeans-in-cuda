# K-Means-Parallel-Computing

![](https://api.visitorbadge.io/api/VisitorHit?user=natasabrisudova&repo=K-Means-Parallel-Computing&countColor=%237B1E7A)

K-Means algorithm parallelized in CUDA

Before trying to execute both sequential and parallel version of
k-means algorithm it is important to extract 'datapoints.zip' in the same 
folder of the codes.

The parameters will be initialized according to user choices 
after the execution start.

If you want to visualize the results you need to create in the same folder 
of the code a directory named 'outdir', execute your simulation and then 
take the file 'clusters.csv' from 'outdir', which 
must be used in the python script 'cluster_displayer.py'.

sequential compiler line on linux:
g++ -o kmeans_sequential kmeans_sequential.cpp

parallel compiler line on linux:
nvcc kmeans_parallel.cu -o kmeans_parallel


by Brisudova Natasa, Graziuso Natalia, Lazzeri Sean Cesare
