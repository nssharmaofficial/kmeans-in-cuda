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
void kMeansCentroidUpdate(float* h_datapoints, int* h_clust_assn, float* h_centroids, int* h_clust_sizes, int N){  

    float clust_datapoint_sums[2*K] = {0};

    for(int j=0; j<N; ++j)
    {
        // clust_id represents a cluster from 1...K
        int clust_id = h_clust_assn[j];
        clust_datapoint_sums[2*clust_id] += h_datapoints[2*j];
        clust_datapoint_sums[2*clust_id+1] += h_datapoints[2*j+1];
        h_clust_sizes[clust_id] += 1;
    }

	//Division by size (arithmetic mean) to compute the actual centroids
	for(int idx = 0; idx < K; idx++){
		if(h_clust_sizes[idx])
		{
			h_centroids[2*idx] = clust_datapoint_sums[2*idx]/h_clust_sizes[idx]; 
			h_centroids[2*idx+1] = clust_datapoint_sums[2*idx+1]/h_clust_sizes[idx]; 
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
	for (int i=0; i<K; i++){
		int temp = (N/K);
		int idx_r = rand()%temp;
		h_centroids[2*i]= h_datapoints[(i*temp +idx_r)];
		h_centroids[2*i+1] = h_datapoints[(i*temp +idx_r)+1];
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
    cout << "Number (int) of points you want to analyze (100, 1000, 10000, 100000, 1000000):\n";
    std::cin >> *num;
    int n = *num;
    switch (n) 
    {
        case 100: *infile_name = "points_100.txt";
        break;
	case 500: *infile_name = "points_500.txt";
        break;
        case 1000: *infile_name = "points_1_000.txt";
        break;
        case 10000: *infile_name = "points_10_000.txt";
        break;
	case 50000: *infile_name = "points_50_000.txt";
        break;
        case 100000: *infile_name = "points_100_000.txt";
        break;
	case 250000: *infile_name = "points_250_000.txt";
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
  
	// ROI CP0 - transferring data from CPU to GPU
	auto start_ROI_cp0 = high_resolution_clock::now();
	cudaMemcpy(d_centroids, h_centroids, D*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints, h_datapoints, D*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes, h_clust_sizes, K*sizeof(int), cudaMemcpyHostToDevice);
	auto stop_ROI_cp0 = high_resolution_clock::now();
    
  	// get and print the time of ROI CP0
	auto duration_ROI_cp0 = duration_cast<microseconds>(stop_ROI_cp0 - start_ROI_cp0);
	float temp = duration_ROI_cp0.count();
	cout << "Time taken by transfering centroids, datapoints and cluster's sizes from host to device is : "<< temp << " microseconds" << endl;

	int cur_iter = 0;

	float time_assignments = 0;         // total time of ROI ASSIGNMENT
	float time_copy= 0;                 // total time of ROI CP
	float time_copy_2= 0;               // total time of ROI CP2
  
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
		
		// ROI CP - copying data (new centroids and cluster assignment) from GPU to CPU
		auto start_ROI_cp = high_resolution_clock::now();
		cudaMemcpy(h_centroids, d_centroids, D*K*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clust_assn, d_clust_assn, N*sizeof(int), cudaMemcpyDeviceToHost);
		auto stop_ROI_cp = high_resolution_clock::now();
        
   		// get the time of ROI CP
		auto duration_ROI_cp = duration_cast<microseconds>(stop_ROI_cp - start_ROI_cp);
		float temp_ROI_cp = duration_ROI_cp.count();
		time_copy  = time_copy + temp_ROI_cp;
		
		//reset centroids and cluster sizes (will be updated in the next kernel)
		memset(h_centroids, 0.0, D*K*sizeof(float));
		memset(h_clust_sizes, 0, K*sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate(h_datapoints, h_clust_assn, h_centroids, h_clust_sizes, N);
    
    		// ROI CP2 - transfering data from CPU to GPU
    		auto start_ROI_cp2 = high_resolution_clock::now();
    		cudaMemcpy(d_centroids, h_centroids, D*K*sizeof(float), cudaMemcpyHostToDevice);
    		auto stop_ROI_cp2 = high_resolution_clock::now();
    
    		// get the time of ROI CP2
		auto duration_ROI_cp2 = duration_cast<microseconds>(stop_ROI_cp2 - start_ROI_cp2);
		float temp_ROI_cp2 = duration_ROI_cp2.count();
		time_copy_2 = time_copy_2 + temp_ROI_cp2;
		
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
	
 	// print the average time of ROI CP during each iteration 
	time_copy= time_copy/MAX_ITER;
	cout << "Time taken by transfering centroids and assignments from the device to the host: "<< time_copy << " microseconds" << endl;

 	// print the average time of ROI CP during each iteration 
	time_copy_2 = time_copy_2/MAX_ITER;
	cout << "Time taken by transfering centroids and assignments from the device to the host: "<< time_copy_2 << " microseconds" << endl;
  
  	// print final centroids
	cout<<"N = "<<N<<",K = "<<K<<", MAX_ITER= "<<MAX_ITER<<".\nThe centroids are:\n";
    	for(int l=0; l<K; l++){
        	cout<<"centroid: " <<l<<": (" <<h_centroids[2*l]<<", "<<h_centroids[2*l+1]<<")"<<endl;
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
