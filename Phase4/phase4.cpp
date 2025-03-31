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

#define DBL_EPS 1e-12
enum InitMethod { RANDOM_SELECTION, RANDOM_PARTITION, MAXIMIN_SELECTION };
enum ValidMethod { CH, SW, D, DB };

// Global variables
string file_name;
int num_cluster, max_iter, num_runs, num_points, dimension;
double threshold;
vector<vector<double> > data;
vector<vector<double> > centers; 
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
            if (fabs(vmax - vmin) < DBL_EPS)
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
            double diff = data[i][j] - mean;
            deviation += diff * diff;
        } 
        deviation = sqrt(deviation / num_points);

        for (int i = 0; i < num_points; i++)
        {
            if (fabs(deviation) < DBL_EPS)
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


// Function to implement the Maximin method for selecting cluster centers
vector<vector<double> > MaximinSelect(const int num_cluster, const vector<vector<double> > &dataset)
{
    vector<vector<double> > centers(num_cluster); 
    vector<int> indices; 

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> distrib(0, num_cluster -1);
    //Choose first Center
    // center_index = distrib(gen); 
    int center_index = 0;
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


//Function that computes the inter-cluster dispersion
double SumDiagonal(const vector<vector<double> >& centers, const vector<vector<double> > &dataset, const vector<int> &assignments)
{
    double Sb = 0.0;
    vector<double> overall_mean(dimension, 0.0);
    vector<int> cluster_sizes(num_cluster, 0);  

    //Find center of dataset
    for (int i = 0; i < num_points; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            overall_mean[j] += dataset[i][j];
        }
        cluster_sizes[assignments[i]]++;
    }

    for (int j = 0; j < dimension; j++)
    {
        overall_mean[j] /= num_points;
    }


    //Calculate distance each clusteroid center and dataset center
    for (int k = 0; k < num_cluster; k++) 
    {
        double dist = SquaredEuclidean(centers[k], overall_mean);
        Sb += cluster_sizes[k] * dist;  
    }

    return Sb;

}

//Function that computes the Calinski-Harabasz index
double CHIndex(const vector<vector<double> > &centers, const vector<vector<double> > &dataset, const vector<int> &assignments, double sse)
{

    double Sb = SumDiagonal(centers, dataset, assignments);
    double Sw = sse;
    double ch_index = (Sb / (num_cluster -1)) / (Sw / (num_points - num_cluster));

    return ch_index;
}


//Function that computes the distances between point pairs
vector<vector<double> > DistanceMatrix(const vector<vector<double> > &dataset) 
{
    vector<vector<double> > distance_matrix(num_points, vector<double>(num_points, 0.0));

    for (int i = 0; i < num_points; i++) 
    {
        for (int j = i + 1; j < num_points; j++) 
        {
            double dist = SquaredEuclidean(dataset[i], dataset[j]);
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist; 
        }
    }
    return distance_matrix;
}

//Function that computes the Silhouette Width index
double SWIndex(const vector<vector<double> > &dataset, const vector<int> &assignments, const vector<vector<double> > &distance_matrix)
{
        vector<double> silhouette_scores(num_points, 0.0);
        double sum = 0.0;

    //Silhouette score for each point
    for (int i = 0; i < num_points; i++) 
    {
        int cluster = assignments[i];
        double a_i = 0.0;
        double b_i = std::numeric_limits<double>::max();
        int cluster_size = 0;

        // Calculate the mean intra-cluster distance (cohesion)
        for (int j = 0; j < num_points; j++) 
        {
            if (assignments[j] == cluster && i != j) 
            {
                a_i += distance_matrix[i][j]; 
                cluster_size++;
            }
        }
            a_i /= cluster_size; 

        // calculate the minimum mean distance to another cluster (separation)
        for (int k = 0; k < num_cluster; k++) 
        {
            // Skip own cluster
            if (k == cluster) 
            {
                continue; 
            }

            double avg_dist = 0.0;
            int cluster_count = 0;
            for (int j = 0; j < num_points; j++) 
            {
                if (assignments[j] == k) 
                {
                    avg_dist += distance_matrix[i][j];
                    cluster_count++;
                }
            }

            // Average inter-cluster distance
            avg_dist /= cluster_count; 
            if (avg_dist < b_i)
            {
                b_i = avg_dist;
            }
        }

        // Compute silhouette score for point i
        if (b_i > a_i) 
        {
            silhouette_scores[i] = (b_i - a_i) / b_i;
        } 
        else if (b_i < a_i) 
        {
            silhouette_scores[i] = (b_i - a_i) / a_i;
        } 
        else 
        {
            silhouette_scores[i] = 0.0;
        }

        sum += silhouette_scores[i];
    }

    return sum / num_points;
    
}

//Function that computes the Dunn index
double DunnIndex(const vector<vector<double> > &dataset, const vector<int> &assignments, const vector<vector<double> > &distance_matrix) 
{
    double w_min_out = std::numeric_limits<double>::max(); 
    double w_max_in = std::numeric_limits<double>::min(); 

    // Compute max intra-cluster distance
    for (int i = 0; i < num_points; i++) 
    {
        for (int j = i + 1; j < num_points; j++) 
        { 
            if (assignments[i] == assignments[j]) 
            { 
                if (w_max_in < distance_matrix[i][j])
                {
                    w_max_in = distance_matrix[i][j];
                }
            }
        }
    }

    // Compute min inter-cluster distance
    for (int i = 0; i < num_points; i++) 
    {
        for (int j = i + 1; j < num_points; j++) 
        { 
            if (assignments[i] != assignments[j]) 
            { 
                if (w_min_out > distance_matrix[i][j])
                {
                    w_min_out = distance_matrix[i][j];
                }
            }
        }
    }

    return w_min_out / w_max_in;
}

//Function that computes the Davies-Bouldin Index
double DBIndex(const vector<vector<double> > &centers, const vector<vector<double> > &dataset, const vector<int> &assignments, const vector<vector<double> > &distance_matrix)
{
    vector<double> cluster_size(num_cluster, 0);

    // Compute dispersions 
    vector<double> dispersions(num_cluster, 0.0);
    for (int i = 0; i < num_points; i++) 
    {
        int cluster = assignments[i];
        double distance = SquaredEuclidean(dataset[i], centers[cluster]);
        dispersions[cluster] += distance;
        cluster_size[cluster]++;
    }

    // Standard deviation
    for (int k = 0; k < num_cluster; k++) 
    {
        dispersions[k] = sqrt(dispersions[k] / cluster_size[k]); 
    }

    // Compute Index
    double db_index = 0.0;
    for (int i = 0; i < num_cluster; i++) 
    {
        double max_ratio = std::numeric_limits<double>::lowest(); 
        for (int j = 0; j < num_cluster; j++) 
        {
            if (i == j)
            {
                continue;
            }
            // Distance between centers of two clusters
            double centroid_dist = sqrt(SquaredEuclidean(centers[i], centers[j])); 
            double db_ij = (dispersions[i] + dispersions[j]) / centroid_dist;
            if (db_ij > max_ratio) 
            {
                max_ratio = db_ij;
            }
        }
        db_index += max_ratio;
    }

    return db_index / num_cluster; 
}

//Function that conducts k-means clustering 
double k_means(const vector<vector <double> > &dataset, InitMethod method, ValidMethod valid_method, const vector<vector<double> > &distance_matrix)
{
    int best_run = -1;
    double best_init_sse = std::numeric_limits<double>::max();
    double best_sse = std::numeric_limits<double>::max();
    int best_iter = std::numeric_limits<int>::max();
    vector<int> assignments;
    vector<double> distances(dataset.size(), 0.0);

    for(int r = 0; r < num_runs; r++)
    {
        double sse = 0.0;
    

        switch(method)
        {
            case RANDOM_SELECTION:
                centers = SelectCluster(num_cluster, dataset);
                assignments = AssignClusters(centers, distances, dataset);
                sse = SSE(distances);
                break;

            case RANDOM_PARTITION:
                assignments = AssignClustersPartition(dataset);
                centers = SelectClusterPartition(assignments, dataset);
                for(int i = 0; i < num_points; i++)
                {
                    int cluster_assignment = assignments[i];
                    sse += SquaredEuclidean(dataset[i], centers[cluster_assignment]);
                }
                break;

            case MAXIMIN_SELECTION:
                centers = MaximinSelect(num_cluster, dataset);
                assignments = AssignClusters(centers, distances, dataset);
                sse = SSE(distances);
                break;
        }

        if (sse < best_init_sse)
        {
            best_init_sse = sse;
        }

        for(int iter = 1; iter < max_iter; iter++)
        {
            //Update the Centers and repeat the process untill threshold
            centers = UpdateClusters(centers, assignments, dataset);
            assignments = AssignClusters(centers, distances, dataset);

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

    
    double index = 0.0;

    switch(valid_method)
    {
        case CH:
            index = CHIndex(centers, dataset, assignments, best_sse);
            cout << index << endl;
            break;

        case SW:
            index = SWIndex(dataset, assignments, distance_matrix);
            cout << index << endl;
            break;

        case D:
            index = DunnIndex(dataset, assignments, distance_matrix);
            cout << index << endl;
            break;
            
        case DB:
            index = DBIndex(centers, dataset, assignments, distance_matrix);
            cout << index << endl;
            break;
    }

    return index;
}


//Function that conducts internal validation to the k-means clustering
void InternalValidation(const vector<vector <double> > &dataset, ValidMethod method, const vector<vector<double> > &distance_matrix)
{
    int kmin = 2;
    int kmax = std::round(sqrt(num_points /2));
    double best_index = std::numeric_limits<double>::lowest();
    double best_index_db = std::numeric_limits<double>::max();
    double index = 0.0;
    double best_cluster = kmin;


    for (int k = kmin; k <= kmax; k++)
    {
        cout << "Number of cluster: " << k << ",";
        num_cluster = k;

        switch(method)
        {
            case CH:
                index = k_means(dataset, MAXIMIN_SELECTION, CH, distance_matrix);
                if (best_index < index)
                {
                    best_index = index;
                    best_cluster = k;
                }
                break;

            case SW:
                index = k_means(dataset, MAXIMIN_SELECTION, SW, distance_matrix);
                if (best_index < index)
                {
                    best_index = index;
                    best_cluster = k;
                }
                break;

            case D:
                index = k_means(dataset, MAXIMIN_SELECTION, D, distance_matrix);
                if (best_index < index)
                {
                    best_index = index;
                    best_cluster = k;
                }
                break;
                
            case DB:
                index = k_means(dataset, MAXIMIN_SELECTION, DB, distance_matrix);
                if (best_index_db > index)
                {
                    best_index_db = index;
                    best_cluster = k;
                }
                break;
        }
    }
    
    if (method == DB)
    {
        cout << "Estimated Number of Clutsers: " << best_cluster << "," << best_index_db << endl;
    }
    else
    {
        cout << "Estimated Number of Clutsers: " << best_cluster << "," << best_index << endl;
    }


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
    if (argc != 5)
    {
        cout << "Not enough arguments" << endl;
        return 1;
    }

    file_name = argv[1];
    max_iter = stoi(argv[2]);
    threshold = stod(argv[3]);
    num_runs = stoi(argv[4]);

    // Validate inputs
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
    vector<vector<double> > distance_matrix = DistanceMatrix(data);


    cout << "Calinski Harabasz, (CH)"  << endl;
    InternalValidation(minmax_data, CH, distance_matrix);


    cout << "Silhouette Width, (SW)" << endl;
    InternalValidation(minmax_data, SW, distance_matrix);
    
    cout << "Dunn, (D)" << endl;
    InternalValidation(minmax_data, D, distance_matrix);

    cout << "Davies Bouldin, (DB)" << endl;
    InternalValidation(minmax_data, DB, distance_matrix);


    return 0;
}