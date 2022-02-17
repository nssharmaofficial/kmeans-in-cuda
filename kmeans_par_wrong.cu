// kMeansCentroidUpdate NOT working properly, 
// there must be some bug!!! 

#include <stdio.h>
#include <time.h>
#include <iostream>     // file-reading
#include <sstream>      // file-reading
#include <fstream>      // file-reading
#include <ctime>     	// for random seeding
#include <chrono>	// for time measuring

using namespace std::chrono;
using namespace std;

#define D 2 		// Dimension of points
#define K 10	        // Number of clusters
#define TPB 32		// Number of threads per block


// Euclidean distance of two 2D points
__device__ float distance(float x1, float y1, float x2, float y2)
{
	return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
}


__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids, int N)
{
	//get idx for this datapoint
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = __FLT_MAX__;
	int closest_centroid = -1;

	for(int c = 0; c < K; ++c)
	{
        /* data points = [x1, y1,...,xn, yn]
            centroids = [c1_x, c1_y,..., ck_x, ck_y]
        */ 
    
     	float dist = distance(d_datapoints[2*idx], d_datapoints[2*idx+1], d_centroids[2*c], d_centroids[2*c+1]);

		// Update of new cluster if it's closer 
		if(dist < min_dist)
		{	
			min_dist = dist;        // update the minimum distance to the current
			closest_centroid = c;   // current closest centroid
		}	
	}
	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx] = closest_centroid;
}

// updating the new centroids according to the mean value of all the assigned data points
__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes, int N)
{

	//get idx of thread at grid level
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int tid = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared 
	//memory so that they can be summed by thread 0 later
	__shared__ float s_datapoints[2 * TPB]; 
	s_datapoints[2*tid]= d_datapoints[2*idx];         // for x coordinates
    	s_datapoints[2*tid+1]= d_datapoints[2*idx+1];     // for y coordinates

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[tid] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the 
	//values within the shared array for the block it is in
	if(tid == 0)
	{
		float *b_clust_datapoint_sums = 0;
       		int *b_clust_sizes = 0;

		for(int p=0; p<K; p++){
			b_clust_datapoint_sums[2*p] = 0.0;
			b_clust_datapoint_sums[2*p+1] = 0.0;
			b_clust_sizes[p] = 0;
		}
		
		// for each thread (point) in the block
		for(int j=0; j<blockDim.x; ++j)
		{
			if(idx+j<N){
                
                		// clust_id represented by the point from clust_assign
				int clust_id = s_clust_assn[j];
                
                		// summation of both of the coordinates within the block for the assigned cluster
				b_clust_datapoint_sums[2*clust_id] += s_datapoints[2*j];        // for x coordinate
				b_clust_datapoint_sums[2*clust_id+1] += s_datapoints[2*j+1];    // for y coordinate
                
               			// count the total number of data points within the cluster
				b_clust_sizes[clust_id] += 1;
			}
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for(int z=0; z < K; ++z)
		{
            		// adding the block centroids to a global centroid for each k centroid
			atomicAdd(&d_centroids[2*z],b_clust_datapoint_sums[2*z]);       // for x coordinate
           		atomicAdd(&d_centroids[2*z+1],b_clust_datapoint_sums[2*z+1]);   // for y coordinate 

            		// counting num of points inside cluster
			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if(idx<K) // maybe for instead of if(c<K)
	{	
		// we are not dividing if there are no points inside the cluster
		if (d_clust_sizes[idx])
		{
			d_centroids[2*idx] = d_centroids[2*idx]/d_clust_sizes[idx]; 
			d_centroids[2*idx+1] = d_centroids[2*idx+1]/d_clust_sizes[idx]; 
		}
	}

}


bool Read_from_file(float *h_datapoints, std::string input_file = "points_100.txt"){
    
	//initalize datapoints
	FILE* file = fopen(input_file.c_str(), "r");

	if(file != NULL){
		int d = 0;
		while ( !feof(file) )
		{
			float x, y;
            
            		// break if you will not find a pair
			if(fscanf(file, "%f %f", &x, &y )!= 2){
				break;
			}
			h_datapoints[2*d] = x;
			h_datapoints[2*d+1] = y;
			d = d + 1;
		}
		fclose(file);
		return 0;

	}else{
		cerr<<"Error during opening file \n";
		return -1;
	}
};

void centroid_init(float* h_datapoints, float* h_centroids, int N){
	//initalize centroids	
	for (int c=0; c<K; c++){
		int temp = (N/K);
		int idx_r = rand()%temp;
		h_centroids[2*c]= h_datapoints[(c*temp +idx_r)];
		h_centroids[2*c+1] = h_datapoints[(c*temp +idx_r)+1];
	}
};

// size is the number of points in the chosen array
void write2csv(float* points, std::string outfile_name, int size)
{  
    std::ofstream outfile;
    outfile.open( outfile_name );
    outfile << "x,y\n";  // name of the columns

    for(int i = 0; i < size; i++){
        outfile << points[2*i] << "," << points[2*i+1] << "\n";
    }
}

/*
For saving to csv file points coordinates and their correspondent cluster
in the format x, y, c
where x, y are the two coordinates and c the relative cluster.

It takes as arguments: the datapoints (of 2*N elem), 
cluster assignment (of N elem), 
name of the output file,
the size (N).
*/
void write2csv_clust(float* points, int* clust_assn, 
                        std::string outfile_name, int size)
{  
    std::ofstream outfile;
    outfile.open( outfile_name );
    outfile << "x,y,c\n";  // name of the columns

    // writing of the coordinates (even are x's, odd are y's) and their relative cluster
    for(int i = 0; i < size; i++){
        outfile << points[2*i] << "," << points[2*i+1] << "," << clust_assn[i] << "\n";
    }
}

void input_user(std::string* infile_name, int* num, int* epochs) 
{
    cout << "Number (int) of points you want to analyze (100, 1000, 10000, 100000):\n";
    std::cin >> *num;
    int n = *num;
    switch (n) 
    {
        case 100: *infile_name = "points_100.txt";
        break;
        case 1000: *infile_name = "points_1_000.txt";
        break;
	case 1024: *infile_name = "points_1024.txt";
        break;
        case 10000: *infile_name = "points_10_000.txt";
        break;
        case 100000: *infile_name = "points_100_000.txt";
	break;
	case 1000000: *infile_name = "points_1_000_000.txt";
        break;
        default: *infile_name = "points_100.txt";
        cout << "Attention: Dataset with " << (n) 
        << " points does not exist!\nThe \"points_100.txt\" dataset will be chosen instead by default :-) ...\n\n";        
        break;
    }
	
    cout << "Please, insert number (int) of epochs for training (in the order of the hundreds is recommended):\n";
    cin >> *epochs;
}

int main()
{
	std::string input_file;
	int N, MAX_ITER;
	input_user(&input_file, &N, &MAX_ITER);

	//allocation of memory on the device 
	float *d_datapoints = 0;
	int *d_clust_assn = 0;
	float *d_centroids = 0;
	int *d_clust_sizes = 0;

	cudaMalloc(&d_datapoints, D*N*sizeof(float));
	cudaMalloc(&d_clust_assn, N*sizeof(int)); 
	cudaMalloc(&d_centroids, D*K*sizeof(float));
	cudaMalloc(&d_clust_sizes,K*sizeof(float));

	// allocation of memory in host
	float *h_centroids = (float*)malloc(D*K*sizeof(float)); 
	float *h_datapoints = (float*)malloc(D*N*sizeof(float));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));
	int *h_clust_assn = (int*)malloc(N*sizeof(int));

	srand(5);
	
	//initialize datapoints
	Read_from_file(h_datapoints, input_file);

	//initialize centroids
	centroid_init(h_datapoints, h_centroids, N);

	for(int c=0; c<K; ++c){
		printf("Initialization of %d centroids: \n", K);
		printf("(%f, %f)\n", h_centroids[2*c], h_centroids[2*c+1]);
	}

	
    	//initialize centroids counter for each clust
    	for(int c = 0; c < K; ++c){
		h_clust_sizes[c] = 0;
	}

  
	Read_from_file(h_datapoints, input_file);


	// ROI 1 - transferring data from CPU to GPU
	auto start_ROI1 = high_resolution_clock::now();
	cudaMemcpy(d_centroids, h_centroids, D*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints, h_datapoints, D*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes, h_clust_sizes, K*sizeof(int), cudaMemcpyHostToDevice);
	auto stop_ROI1 = high_resolution_clock::now();
    
    	// get and print the time of ROI 1
	auto duration_ROI1 = duration_cast<microseconds>(stop_ROI1 - start_ROI1);
	float temp = duration_ROI1.count();
	cout << "Time taken by transfering centroids, datapoints and cluster's sizes from host to device is : "<< temp << " microseconds" << endl;

	int cur_iter = 0;

	float time_assignments = 0;         // total time of ROI2
	float time_copy_by_device = 0;      // total time of ROI4

	// ROI WHILE - while cycle (duration of all epochs)
	auto start_while = high_resolution_clock::now();
	while(cur_iter < MAX_ITER)
	{

		// ROI ASSIGNMENT - cluster assignment
		auto start = high_resolution_clock::now();
		kMeansClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints, d_clust_assn, d_centroids, N);
		auto stop = high_resolution_clock::now();
        
        	// get the time of ROI ASSIGNMENT
		auto duration = duration_cast<microseconds>(stop - start);
		float temp = duration.count();
		time_assignments = time_assignments + temp;
		
		// ROI 2 - copying data (new centroids and cluster assignment) from GPU to CPU
		auto start_ROI2 = high_resolution_clock::now();
		cudaMemcpy(h_centroids, d_centroids, D*K*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clust_assn, d_clust_assn, N*sizeof(int), cudaMemcpyDeviceToHost);
		auto stop_ROI2 = high_resolution_clock::now();
        
        	// get the time of ROI 2
		auto duration_ROI2 = duration_cast<microseconds>(stop_ROI2 - start_ROI2);
		float temp_ROI2 = duration_ROI2.count();
		time_copy_by_device = time_copy_by_device + temp_ROI2;
		
		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids, 0.0, D*K*sizeof(float));
		cudaMemset(d_clust_sizes, 0, K*sizeof(int));

		// centroid update
		kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints, d_clust_assn, d_centroids, d_clust_sizes, N);
		
		cur_iter += 1;
	}
    
	auto stop_while = high_resolution_clock::now();
    
    	// get and print the time of ROI WHILE
	auto duration_while = duration_cast<microseconds>(stop_while - start_while);
	float temp_while = duration_while.count();
	cout << "Time taken by " << MAX_ITER << " iterations is: "<< temp_while << " microseconds" << endl;

    	// print the average time of ROI ASSIGNMENT during each iteration 
	time_assignments = time_assignments/MAX_ITER;
	cout << "Time taken by kMeansClusterAssignment: "<< time_assignments << " microseconds" << endl;
	
    	// print the average time of ROI2 during each iteration 
	time_copy_by_device = time_copy_by_device/MAX_ITER;
	cout << "Time taken by transfering centroids and assignments from the device to the host: "<< time_copy_by_device << " microseconds" << endl;

      	// print final centroids
	cout<<"N = "<<N<<",K = "<<K<<", MAX_ITER= "<<MAX_ITER<<".\nThe centroids are:\n";
    	for(int c=0; c<K; c++){
        	cout<<"centroid: " <<c<<": (" <<h_centroids[2*c]<<", "<<h_centroids[2*c+1]<<")"<<endl;
        }


	// Naming for the output files
	std::string outfile_points = "./outdir/datapoints.csv";
	std::string outfile_centroids = "./outdir/centroids.csv";
	std::string outfile_clust = "./outdir/clusters.csv";

	// Writing to files
	write2csv(h_datapoints, outfile_points, N);
	write2csv(h_centroids, outfile_centroids, K);
	write2csv_clust(h_datapoints, h_clust_assn, outfile_clust, N);
	
	// Freeing memory on device
	cudaFree(d_datapoints);
	cudaFree(d_clust_assn);
	cudaFree(d_centroids);
	cudaFree(d_clust_sizes);

	// Freeing memory on host
	free(h_centroids);
	free(h_datapoints);
	free(h_clust_sizes);
	free(h_clust_assn);

	return 0;
}
