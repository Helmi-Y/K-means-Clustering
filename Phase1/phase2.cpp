/*

1) Can k-means initialized uniformly at random (as above) encounter empty
clusters in any iteration? If yes, explain when and why this can happen. Otherwise,
explain why this is not possible. Include your explanation as a comment at the top of
your main source file. 

I believe that k-means that is initialized uniformly at random may encounter empty 
clusters in an iteration. This can occur if the cluster center that is initialized 
does not have any nearby data points, or if the data points all have a closer cluster
that they can be assigned to. The issue lies in the initial selection of the clusters
as it will be random, and the AssignCluster function will not assign any points to that
cluster. 

In this program, we have created a solution for a cluster that only has a single data 
point. This is done by detecting it, and assigning a new center based on the point with 
the larget error/distanace from its current cluster. However, I don't think that it will
allow clusters that are empty, especially in early iterations. Thus, additional functions
might need to be implemented to select random points to that empty cluster, or adjust the 
center.

2) Consider the case where a data point has more than one nearest center. Most
implementations assign such a point to the nearest center with the smallest index. Why do
you think such a scheme has become a convention in the literature? If, instead, you assign
such a point to a nearest center selected uniformly at random, does the algorithm still
converge? If not, why not? Add this random assignment strategy to your program as an
option and explain its theoretical (not whether or not it appears to work in practice)
convergence behavior in a comment at the top of your main source file. 

I believe that that this scheme has become a convention in the literature because it is 
simple and consistent for the algorithm. It ensures that the assignment of points is 
not random, and it will be easy to implement for the algorithm to converge. There won't
any ambiguity as the convergence is using a consistent assignment. 

If instead, we assign the point to a nearest center selected unifromly at random, I believe 
that the algorithm will still converge. In theory, the cluster centers will still stabilize
after many iterations as the random assignment won't stop the algorithm from 
reassigning data points and the points will most likely be assigned to centers that are
more stable in the end. Since we are minimzing the sum of squared errors, randomly 
assigning the points that are equal to more than one center might temporarily increase
the SSE, but it will eventually converge as long as the centers will be updated after
the assignment of clusters that will reduce the SSE. 

However, the randomness might lead to slower convergence time 
for the algorithm as it can be inefficient and incosistent. This incosistency will 
cause the algorithm to take longer to stablizie. It can also not guarantee convergence 
as the randomness might lead to situations where the data points are in a cycle 
and are assigned to the same clusters as more iterationsoccur. 


*/



// This program follows the Google C++ Style Guide:
// https://google.github.io/styleguide/cppguide.html

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>

// Avoid 'using namespace std'
using std::cout;
using std::endl;
using std::ifstream;
using std::stod;
using std::stoi;
using std::string;
using std::vector;

// Global variables
string file_name;
int num_cluster, max_iter, num_runs, num_points, dimension;
double threshold;
vector<vector<double> > data;
vector<vector<double> > centers; 

// Function to read file and store data
void ReadFile(const string &file_name)
{
    ifstream file;

    file.open(file_name);

    if (!file.is_open()) 
    {
        cout << "Error: Could not open file " << file_name << endl;
        return;
    }

    if (file.is_open())
    {
        file >> num_points >> dimension;
    }

    data.resize(num_points, vector<double>(dimension));

    for (int i = 0; i < num_points; i++)
    {

        for (int j = 0; j < dimension; j++)
        {
            file >> data[i][j];
        }
    }

    file.close();
}

// Function to check if random index has been selected
bool IndexCheck(int rand_index, const vector<int> &indices)
{
    for (int i = 0; i < indices.size(); i++)
    {
        if (indices[i] == rand_index)
            return false;
    }
    return true;
}

// Function to print the selected clusters
void PrintPoints(const vector<int> &indices)
{
    for (int i = 0; i < indices.size(); i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            cout << data[indices[i]][j] << " ";
        }
        cout << endl;
    }
}

// Function to randomly select the cluster centers
vector<vector<double> > SelectCluster(const int num_cluster)
{

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(0, data.size() -1);

    int rand_index;
    vector<int> indices;
    vector<vector<double> >centers; 

    for (int i = 0; i < num_cluster; i++)
    {
        do
        {
            rand_index = distrib(gen);    
        } while (!IndexCheck(rand_index, indices));

        indices.push_back(rand_index);
        centers.push_back(data[rand_index]);
    }

    return centers; 
}

// Function to calculate the Squared Euclidean Distance
double SquaredEuclidean (const vector<double> &data1, const vector<double> &data2)
{
    double distance = 0.0;

    // Sum the squared differences between each dimension of the points
    for (int d = 0; d < dimension; d++)
    {
        double difference = data1[d] - data2[d];
        distance += difference * difference;
    }
    return distance; 
}

// Function to assign data points into its closest cluster center
vector<int> AssignClusters(const vector<vector <double> > &centers)
{
   vector<int> assignments(num_points); 

    // Calculate the distance of each point to all cluster centers and assign the closest one
    for(int i = 0; i < num_points; i++)
    {
        double lowest_distance = 999999;
        int closest_center;

        for(int k = 0; k < num_cluster; k++)
        {
            double distance = SquaredEuclidean(data[i], centers[k]);
            if (distance < lowest_distance)
            {
                lowest_distance = distance;
                closest_center = k;
            }

        }

    //Assign data point to closest cluster
       assignments[i] = closest_center;
    }

    return assignments;
}


// Function to calculate the Sum of Squared Errors for each point with their respective clusters
double SSE(const vector<vector<double> >&centers, const vector<int>&assignments)
{
    double sse = 0.0;

    //Caluclate SSE for each point based on its cluster assignment
    for(int i = 0; i < num_points; i++)
    {
        int cluster_assignment = assignments[i];
        sse += SquaredEuclidean(data[i], centers[cluster_assignment]);
    }

    return sse; 
}

// Function to update the cluster assignments
vector<vector<double> >UpdateClusters(const vector<vector<double> >&centers, const vector<int>&assignments)
{
    vector<vector<double> > new_centers(num_cluster, vector<double>(dimension, 0.0));
    vector<vector<double> > cluster_sum(num_cluster, vector<double>(dimension, 0.0));
    vector<double> cluster_size(num_cluster, 0);

    // Update the cluster centers by summing the points in each cluster
    for(int i = 0; i < num_points; i++)
    {

        int cluster_assignment = assignments[i];
        cluster_size[cluster_assignment]++;
        for (int d = 0; d < dimension; d++)
        {
            cluster_sum[cluster_assignment][d] += data[i][d];
        }
    }


    // Assign new centers for each cluster
    for(int k = 0; k < num_cluster; k++)
    {
        if (cluster_size[k] > 1)
        {
            for(int d = 0; d < dimension; d++)
            {
                // Mean center 
                new_centers[k][d] = cluster_sum[k][d] / cluster_size[k];
            }
        }
        else
        {
            //Singleton cluster issue
            cout << "A singleton cluster has been detected" << endl;
            int max_error_point = -1;
            double max_error = -1;

            for (int i = 0; i < num_points; i++)
            {
                //Find the point with the most error (distance) from current center
                if (assignments[i] != k)
                {
                    double error = SquaredEuclidean(data[i], centers[k]);
                    if (error > max_error)
                    {
                        max_error = error;
                        max_error_point = i;
                    }
                }
            }

            // Set that point to be the center of the singleton cluster
            new_centers[k] = data[max_error_point];
        }
    }

    return new_centers;

}

int main(int argc, char *argv[])
{
    // <F> <K> <I> <T> <R>, where
    //  F: name of the data file
    //  K: number of clusters (positive integer greater than one)
    //  I: maximum number of iterations (positive integer)
    //  T: convergence threshold (non-negative real)
    //  R: number of runs (positive integer)

    // Valide number of arguments
    if (argc != 6)
    {
        cout << "Not enough arguments" << endl;
        return 1;
    }

    file_name = argv[1];
    num_cluster = stoi(argv[2]);
    max_iter = stoi(argv[3]);
    threshold = stod(argv[4]);
    num_runs = stoi(argv[5]);

    // Validate inputs
    if (num_cluster <= 1)
    {
        cout << "Number of clusters " << num_cluster << " must be greater than 1" << endl;
        return 0;
    }

    if (max_iter <= 0)
    {
        cout << "Max iteration " << max_iter << " must be greater than 0" << endl;
        return 0;
    }

    if (threshold < 0)
    {
        cout << "Threshold " << threshold << " must be greater than or equal to 0" << endl;
        return 0;
    }

    if (num_runs <= 0)
    {
        cout << "Number of runs " << num_runs << " must be greater than 0" << endl;
        return 0;
    }

    // Read data 
    ReadFile(file_name);

    int best_run = -1;
    double best_sse = 999999;

    for(int r = 0; r < num_runs; r++)
    {
        cout << "Run " << r + 1 << endl;
        cout << "------" << endl;

        //Select initial centers, assign points to its clutsers and calculate it's errors
        centers = SelectCluster(num_cluster);
        vector<int> assignments = AssignClusters(centers);
        double sse = SSE(centers, assignments);
        cout << "Iteration 1: SSE = " << sse << endl;

        for(int iter = 1; iter < max_iter; iter++)
        {
            //Update the Centers and repeat the process untill threshold
            centers = UpdateClusters(centers, assignments);
            assignments = AssignClusters(centers);

            double new_sse = SSE(centers, assignments);
            cout << "Iteration " << iter + 1 << ": SSE = " << new_sse << endl;

            if (new_sse < best_sse)
            {
                best_run = r;
                best_sse = new_sse;
            }
            if((sse - new_sse) / sse < threshold)
            {
                break;
            }

            sse = new_sse;
        }
    }

    cout << endl;
    cout << "Best Run: " << best_run + 1 << ": SSE = " << best_sse << endl;


    return 0;
}