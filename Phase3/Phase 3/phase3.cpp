/*
How does the random partition method compare against 
the random selection method in theory?

The Random Partition Method has each data point be assigned to one of the K clusters.
The inital cluster centers are then computed by calculating the mean of the points
in each cluster. In theory, this method allows all clusters to intially have at least 
one point. However, it may lead to ineffective initial centroids if the random assignment
is not balanced.

The Random Selection method randomly choses the initial centeres from the dataset. Due to
its random nature, some clusters may be closer together, which might cause convergence
to be slower. In theory, although it allows us to skip the need of calculating initial
means, it can lead to bad initial centers which might lead to higher intial SSE. 

The Random Partition method gives a more balanced initialization approach. Each cluster
will start with at least one point and they are all calculated. The random initialization 
of the Random Selection method can often lead to varying results, as it can pick poor starting
centers. In theory, Random Partition should result it lower initial SSE compared to random selection,
but it won't necesarrily cause a more efficient program in the long run. I think that the best
final SSE and the best number of iterations would more or less be similar. I believe that both
methods will still need multiple runs to be effective, but Random Partition should have a better 
intial SSE. 
*/



// This program follows the Google C++ Style Guide:
// https://google.github.io/styleguide/cppguide.html

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <random>
#include <cmath>

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
vector<vector<double> > centers_selection; 
vector<vector<double> > centers_partition; 
vector<vector<double> > centers_maximin; 

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
vector<vector<double> > SelectCluster(const int num_cluster, const vector<vector <double> > &dataset)
{

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(0, dataset.size() -1);

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
        centers.push_back(dataset[rand_index]);
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
vector<int> AssignClusters(const vector<vector <double> > &centers, vector<double> &distances, const vector<vector <double> > &dataset)
{
   vector<int> assignments(num_points); 

    // Calculate the distance of each point to all cluster centers and assign the closest one
    for(int i = 0; i < num_points; i++)
    {
        double lowest_distance = std::numeric_limits<double>::max();
        int closest_center;

        for(int k = 0; k < num_cluster; k++)
        {
            double distance = SquaredEuclidean(dataset[i], centers[k]);
            if (distance < lowest_distance)
            {
                lowest_distance = distance;
                closest_center = k;
            }

        }

    //Assign data point to closest cluster
       assignments[i] = closest_center;
       distances[i] = lowest_distance;
    }

    return assignments;
}


// Function to calculate the Sum of Squared Errors for each point with their respective clusters
double SSE(const vector<double>& distances)
{
    double sse = 0.0;

    //Caluclate SSE for each point based on its cluster assignment
    for(int i = 0; i < num_points; i++)
    {
        sse += distances[i];
    }

    return sse; 
}

// Function to update the cluster assignments
vector<vector<double> >UpdateClusters(const vector<vector<double> >&centers, const vector<int>&assignments, const vector<vector <double> > &dataset)
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
            cluster_sum[cluster_assignment][d] += dataset[i][d];
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
            // cout << "A singleton cluster has been detected" << endl;
            int max_error_point = -1;
            double max_error = -1;

            for (int i = 0; i < num_points; i++)
            {
                //Find the point with the most error (distance) from current center
                if (assignments[i] != k)
                {
                    double error = SquaredEuclidean(dataset[i], centers[k]);
                    if (error > max_error)
                    {
                        max_error = error;
                        max_error_point = i;
                    }
                }
            }

            // Set that point to be the center of the singleton cluster
            new_centers[k] = dataset[max_error_point];
        }
    }

    return new_centers;

}

//Function to normalize dataset using MinMax normalization
 vector<vector<double> > MinMax()
{
    vector<vector<double> > normalized_data(num_points, vector<double>(dimension));

    for (int j = 0; j < dimension; j++)
    {
        double vmax = std::numeric_limits<double>::lowest();
        double vmin = std::numeric_limits<double>::max();


        for (int i = 0; i < num_points; i++)
        {
            if (data[i][j] < vmin)
            {
                vmin = data[i][j];
            }
            if (data[i][j] > vmax)
            {
                vmax = data[i][j];
            }

        }

        for (int i = 0; i < num_points; i++)
        {
            if (vmax == vmin)
            {
                normalized_data[i][j] = 0;  
            }
            else
            {
                normalized_data[i][j] = (data[i][j] - vmin) / (vmax - vmin);
            }
        }
    }
    return normalized_data;

}

//Function to normalize dataset using ZScore normalization
 vector<vector<double> > ZScore()
 {
    vector<vector<double> > zscore_data(num_points, vector<double>(dimension));

    for (int j = 0; j < dimension; j++)
    {
        double mean = 0.0;
        double deviation = 0.0;
        for (int i = 0; i < num_points; i++)
        {
            mean += data[i][j];
        }
        mean /= num_points;


        for (int i = 0; i < num_points; i++)
        {
            deviation += pow(data[i][j] - mean, 2);
        } 
        deviation = sqrt(deviation / num_points);

        for (int i = 0; i < num_points; i++)
        {
            if (deviation == 0)
            {
                zscore_data[i][j] = 0;  
            }
            else
            {
                zscore_data[i][j] = (data[i][j] - mean) / deviation;
            }
        }
    }

    return zscore_data;
 }


//Function that uses Random Selection for initialization
void Random_Selection(const vector<vector <double> > &dataset)
{
    int best_run = -1;
    double best_init_sse = std::numeric_limits<double>::max();
    double best_sse = std::numeric_limits<double>::max();
    int best_iter = std::numeric_limits<int>::max();

    for(int r = 0; r < num_runs; r++)
    {
        //Select initial centers, assign points to its clutsers and calculate it's errors
        centers_selection = SelectCluster(num_cluster, dataset);
        vector<double> distances(dataset.size(), 0.0);
        vector<int> assignments = AssignClusters(centers_selection, distances, dataset);
        double sse = SSE(distances);
        if (sse < best_init_sse)
        {
            best_init_sse = sse;
        }

        for(int iter = 1; iter < max_iter; iter++)
        {
            //Update the Centers and repeat the process untill threshold
            centers_selection = UpdateClusters(centers_selection, assignments, dataset);
            assignments = AssignClusters(centers_selection, distances, dataset);

            double new_sse = SSE(distances);

            if (new_sse < best_sse)
            {
                best_run = r;
                best_sse = new_sse;
            }
            if((sse - new_sse) / sse < threshold)
            {
                if (iter < best_iter)
                {
                    best_iter = iter;
                }
                break;
            }

            sse = new_sse;
        }
    }

    cout << "Random Selection:, ";
    cout << "Run: " << best_run + 1 << " SSE = " << best_sse;
    cout << ", SSE: " << best_init_sse;
    cout << ", " << best_iter + 1 <<  endl;
}



// Function to assign each point to a cluster selected uniformly at random
vector<int> AssignClustersPartition(const vector<vector <double> > &dataset)
{
    vector<int> assignments(num_points); 
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(0, num_cluster -1);
    for (int i = 0; i < num_points; i++)
    {
        int rand_index = distrib(gen);
        assignments[i] = rand_index;
    }

    return assignments;
}

// Function to take the centroids of the initial clusters as the initial centers 
vector<vector<double> > SelectClusterPartition(vector<int> assignments, const vector<vector <double> > &dataset)
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
            cluster_sum[cluster_assignment][d] += dataset[i][d];
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
    }

    return new_centers;
}


//Function that uses Random Partition for initialization
void Random_Partition(const vector<vector <double> > &dataset)
{
    int best_run = -1;
    double best_init_sse = std::numeric_limits<double>::max();
    double best_sse = std::numeric_limits<double>::max();
    int best_iter = std::numeric_limits<int>::max();

    for(int r = 0; r < num_runs; r++)
    {
        //Assign Each Point to a Cluster
        vector<int> assignments_partition = AssignClustersPartition(dataset);
        //Take Centroids of intial cluster as initial centers
        centers_partition = SelectClusterPartition(assignments_partition, dataset);
        vector<double> distances(dataset.size(), 0.0);
        double sse = 0.0;

        //Caluclate SSE for each point based on its cluster assignment
        for(int i = 0; i < num_points; i++)
        {
            int cluster_assignment = assignments_partition[i];
            sse += SquaredEuclidean(dataset[i], centers_partition[cluster_assignment]);
        }

        if (sse < best_init_sse)
        {
            best_init_sse = sse;
        }

        for(int iter = 1; iter < max_iter; iter++)
        {
            //Update the Centers and repeat the process untill threshold
            centers_partition = UpdateClusters(centers_partition, assignments_partition, dataset);
            assignments_partition = AssignClusters(centers_partition, distances, dataset);

            double new_sse = SSE(distances);

            if (new_sse < best_sse)
            {
                best_run = r;
                best_sse = new_sse;
            }
            if((sse - new_sse) / sse < threshold)
            {
                if (iter < best_iter)
                {
                    best_iter = iter;
                }
                break;
            }

            sse = new_sse;
        }
    }

    cout << "Random Partition:, ";
    cout << "Run: " << best_run + 1 << " SSE = " << best_sse;
    cout << ", SSE: " << best_init_sse;
    cout << ", " << best_iter + 1 <<  endl;
}

// Function to implement the Maximin method for selecting cluster centers
vector<vector<double> > MaximinSelect(const int num_cluster, const vector<vector<double> > &dataset)
{
    vector<vector<double> > centers(num_cluster); 
    vector<int> indices; 

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(0, num_cluster -1);
    //Choose first Center
    int center_index = distrib(gen); 
    //center_index = 0;
    centers[0] = dataset[center_index]; 
    indices.push_back(center_index);

    double max_distance = std::numeric_limits<double>::lowest();

    for (int i = 1; i < num_cluster ;i++)
    {
        center_index = -1;
        double max_distance_centers = std::numeric_limits<double>::lowest();

        for (int j = 0; j < dataset.size(); j++)
        {
            double min_distance = std::numeric_limits<double>::max();
            for (int k = 0; k < indices.size(); k++)
            {
                double distance = SquaredEuclidean(dataset[j], centers[k]);
                //Find closest center from point
                if (distance < min_distance) 
                {
                    min_distance = distance; 
                }
            }
            //See if point has the largest distance from the centers
            if (min_distance > max_distance_centers)
            {
                max_distance_centers = min_distance;
                center_index = j;
            }
        }
        centers[i] = dataset[center_index];
        indices.push_back(center_index);
    }
    return centers; 
}

//Function that uses Maximin for initialization
void Maximin(const vector<vector <double> > &dataset)
{
    int best_run = -1;
    double best_init_sse = std::numeric_limits<double>::max();
    double best_sse = std::numeric_limits<double>::max();
    int best_iter = std::numeric_limits<int>::max();

    for(int r = 0; r < num_runs; r++)
    {
        // Select initial centers using the Maximin method
        centers_maximin = MaximinSelect(num_cluster, dataset);
        vector<double> distances(dataset.size(), 0.0);
        vector<int> assignments = AssignClusters(centers_maximin, distances, dataset);
        double sse = SSE(distances);
        if (sse < best_init_sse)
        {
            best_init_sse = sse;
        }

        for(int iter = 1; iter < max_iter; iter++)
        {
            //Update the Centers and repeat the process untill threshold
            centers_maximin = UpdateClusters(centers_maximin, assignments, dataset);
            assignments = AssignClusters(centers_maximin, distances, dataset);

            double new_sse = SSE(distances);

            if (new_sse < best_sse)
            {
                best_run = r;
                best_sse = new_sse;
            }
            if((sse - new_sse) / sse < threshold)
            {
                if (iter < best_iter)
                {
                    best_iter = iter;
                }
                break;
            }

            sse = new_sse;
        }
    }

    cout << "MaxiMin Method:, ";
    cout << "Run: " << best_run + 1 << " SSE = " << best_sse;
    cout << ", SSE: " << best_init_sse;
    cout << ", " << best_iter + 1 <<  endl;
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
    vector<vector<double> > minmax_data(num_points, vector<double>(dimension));
    minmax_data = MinMax();

    vector<vector<double> > zscore_data(num_points, vector<double>(dimension));
    zscore_data = ZScore();


    cout << "MinMax normalization, ";
    cout << "Best Run and SSE, ";
    cout << "Best Initial SSE, ";
    cout << "Best # of Iterations" << endl;
    Random_Selection(minmax_data);
    Random_Partition(minmax_data);
    Maximin(minmax_data);

    cout << "ZScore normalization, ";
    cout << "Best Run and SSE, ";
    cout << "Best Initial SSE, ";
    cout << "Best # of Iterations" << endl;
    Random_Selection(zscore_data);
    Random_Partition(zscore_data);
    Maximin(zscore_data);


    return 0;
}