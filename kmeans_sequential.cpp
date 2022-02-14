#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>   // file-reading
#include <sstream>    // file-reading
#include <fstream>    // file-reading
#include <ctime>      // for random seeding
#include <chrono>     // for time measuring

using namespace std::chrono;
using namespace std;

#define D 2   // Dimension of points


// Euclidean distance of two 2D points
float distance(float x1, float y1, float x2, float y2)
{
	return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
}


// Find the closest centroid to each of N datapoints for each cluster K
void kMeansClusterAssignment(float* datapoints, int* clust_assn, float* centroids, int N, int K) 
{
    for(int idx=0; idx<N; idx++)
      
	{      
        float min_dist = __FLT_MAX__;
        int closest_centroid = -1;
      
        // distance of one point from datapoints and centroid of each cluster
        for(int c = 0; c < K; ++c)
        {
            /* datap oints = [x1, y1,...,xn, yn]
               centroids = [c1_x, c1_y,..., ck_x, ck_y]
            */ 
            float dist = distance(datapoints[2*idx], datapoints[2*idx+1], centroids[2*c], centroids[2*c+1]);

            // update of new cluster if it's closer 
            if(dist < min_dist)
            {	
                min_dist = dist;          // update the minimum distance to the current
                closest_centroid = c;     // current closest centroid
            } 
        }
        // assign the cluster to that point after iteration through all the clusters
        clust_assn[idx] = closest_centroid;
    }	
}

// updating the new centroids according to the mean value of all the assigned data points
void kMeansCentroidUpdate(float* datapoints, int* clust_assn, float* centroids, int* clust_sizes, int N, int K)

{  
    float clust_datapoint_sums[2*K] = {0};
    for(int j=0; j<N; ++j)
    {
        // clust_id represents a number of a cluster (0...K)
        int clust_id = clust_assn[j];
      
        // summation of both of the coordinates within each cluster
        clust_datapoint_sums[2*clust_id] += datapoints[2*j];          // for x coordinates
        clust_datapoint_sums[2*clust_id+1] += datapoints[2*j+1];      // for y coordinates
      
        // count the total number of data points within each cluster
        clust_sizes[clust_id] += 1;
    }

	// arithmetic mean to get the current updated centroid for each cluster
	for(int idx = 0; idx < K; idx++){
		if(clust_sizes[idx]) // to avoid division by zero
		{
			centroids[2*idx] = clust_datapoint_sums[2*idx]/clust_sizes[idx];      // new x coordinate of updated centroid
			centroids[2*idx+1] = clust_datapoint_sums[2*idx+1]/clust_sizes[idx];  // new y coordinate of updated centroid
		}	
	}

}

bool Read_from_file(float* datapoints, std::string input_file = "points_100.txt"){
	//initalize datapoints
	
	FILE* file = fopen(input_file.c_str(), "r");
	if(file != NULL){
        cout <<"The initial points are: \n";
		int d = 0;
        while ( !feof(file) )
		{
			float x, y;
            
            // break if you will not find a pair
			if(fscanf(file, "%f %f", &x, &y )!= 2){
				break;
			}
            datapoints[2*d] = x;
			datapoints[2*d+1] = y;
			d = d + 1;
		}
        fclose(file);
		return 0;

	}else{
		cerr<<"Error during opening file \n";
		return -1;
	}
};

// centroid initialization
bool centroid_init(float* centroids, int k){
	
    // save the initialized centroids in a file
	std::string input_centr_file;

	if(k==8){
		input_centr_file = "init_centroids_8.txt";
	}else if(k==10){
		input_centr_file = "init_centroids_10.txt";
	}else if(k==20){
		input_centr_file = "init_centroids_20.txt";
	}else if(k==50){
		input_centr_file = "init_centroids_50.txt";
	}
    
	FILE* file = fopen(input_centr_file.c_str(), "r");
	if(file != NULL){
		int d = 0;
        while ( !feof(file) )
		{
			float x, y;
            if(fscanf(file, "%f %f", &x, &y )!= 2){
				break;
		    }
            centroids[2*d] = x;
			centroids[2*d+1] = y;
			d = d + 1;
		}
		fclose(file);
		return 0;

	}else{
		cerr<<"Error during opening file \n";
		return -1;
	}
};

// size is the number of points in the chosen array, 
void write2csv(float* points, std::string outfile_name, int size)
{  
    std::ofstream outfile;
    outfile.open(outfile_name);
    outfile << "x,y\n";  // name of the columns

    for(int i = 0; i < size; i++){
        outfile << points[2*i] << "," << points[2*i+1] << "\n";
    }
}

/*
For saving to csv file points coordinates and their
correspondent cluster in the format x, y, c
where x, y are the two coordinates and c the relative cluster.

It takes as arguments: 
the datapoints (of 2*N elem), 
cluster assignment (of N elem), 
name of the output file,
the size (N).
*/
void write2csv_clust(float* points, int* clust_assn, std::string outfile_name, int size)
{  
    std::ofstream outfile;
    outfile.open(outfile_name);
    outfile << "x,y,c\n";  // name of the columns

	// writing of the coordinates (even are x's, odd are y's) and their relative cluster.
    for(int i = 0; i < size; i++){
        outfile << points[2*i] << "," << points[2*i+1] << "," << clust_assn[i] << "\n";
    }
}

// user can define the number of: data points (N), epochs and clusters
void input_user(std::string* infile_name, int* num, int* k, int* epochs) 
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
        default: *infile_name = "points_100.txt";
        cout << "Attention: Dataset with " << (n) 
        << " points does not exist!\nThe \"points_100.txt\" dataset will be chosen instead by default.\n\n";        
        break;
    }
	
    cout << "Please, insert number (int) of epochs for training (in the order of the hundreds is recommended):\n";
    cin >> *epochs;
	
    cout << "Please, insert the number (int) of the k clusters (8 - 10 - 20 - 50):\n";
    cin >> *k;
}


int main()
{
	std::string input_file;
	int N, K, MAX_ITER;
	input_user(&input_file, &N, &K, &MAX_ITER);

	//allocate memory 
	float datapoints[D*N] = {0};
	int clust_assn[N] = {0};
	float centroids[D*N]= {0};
	int clust_sizes[K] = {0};
	
	srand(5);

	//initialize centroids
	centroid_init(centroids, K);
	for(int c=0; c<K; ++c){
		printf("Initialization of %d centroids: \n", K);
		printf("(%f, %f)\n", centroids[2*c], centroids[2*c+1]);
	}

    //initialize datapoints
	Read_from_file(datapoints, input_file);

	int cur_iter = 0;
	float time_assignments = 0;     
	float time_update = 0;

	// ROI1 - while cycle (durations of all epochs)
	auto start_while = high_resolution_clock::now();
	while(cur_iter < MAX_ITER)
	{
		// ROI 2 - cluster assignment
		auto start = high_resolution_clock::now();
		kMeansClusterAssignment(datapoints, clust_assn, centroids, N, K);
		auto stop = high_resolution_clock::now();
        
        // get the time of ROI 2
		auto duration = duration_cast<microseconds>(stop - start);
		float temp = duration.count();
		time_assignments = time_assignments + temp;
  
		// initialize back to zero
		for(int p=0; p<K; p++){
			clust_sizes[p] = 0;
		}
        centroids[2*K] = {0.0};

        // ROI 3 missing in this case because we dont allocate memory
        
		// ROI 4 - centroid update kernel 
		auto start2 = high_resolution_clock::now();
		kMeansCentroidUpdate(datapoints, clust_assn, centroids, clust_sizes, N, K);
		auto stop2 = high_resolution_clock::now();

        // get the time of ROI 4
		auto duration2 = duration_cast<microseconds>(stop2 - start2);
		float temp2 = duration2.count();
		time_update = time_update + temp2;
		
		cur_iter += 1;
	}
    
	auto stop_while = high_resolution_clock::now();
    
    // get the time of ROI 1 
	auto duration_while = duration_cast<microseconds>(stop_while - start_while);
	float temp = duration_while.count();
	cout << "Time taken by " << MAX_ITER << " iterations is: "<< temp << " microseconds" << endl;

    // the average time of ROI2 during each iteration    
	time_assignments = time_assignments/MAX_ITER;
	cout << "Time taken by kMeansClusterAssignment: "<< time_assignments << " microseconds" << endl;
    
    // the average time of ROI4 during each iteration
	time_update = time_update/MAX_ITER;	
	cout << "Time taken by kMeansCentroidUpdate: "<< time_update << " microseconds" << endl;
  
    // print final centroids
	cout<<"N = "<<N<<",K = "<<K<<", MAX_ITER= "<<MAX_ITER<<".\nThe centroids are:\n";
    for(int l=0; l<K; l++){
        cout<<"centroid: " <<l<<": (" <<centroids[2*l]<<", "<<centroids[2*l+1]<<")"<<endl;
    }

	// Naming for the output files
	std::string outfile_points = "./outdir/datapoints.csv";
	std::string outfile_centroids = "./outdir/centroids.csv";
	std::string outfile_clust = "./outdir/clusters.csv";

	// Writing to files
	write2csv(datapoints, outfile_points, N);
	write2csv(centroids, outfile_centroids, K);
	write2csv_clust(datapoints, clust_assn, outfile_clust, N);
	

	return 0;
}
