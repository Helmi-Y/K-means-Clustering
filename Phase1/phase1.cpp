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

// Function to read file and store data
void ReadFile(const string &file_name)
{
    ifstream file;

    file.open(file_name);

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
void SelectCluster(int num_cluster)
{
    srand(time(0));
    vector<int> indices;
    int rand_index;
    for (int i = 0; i < num_cluster; i++)
    {
        do
        {
            std::random_device rd;  // a seed source for the random number engine
            std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
            std::uniform_int_distribution<> distrib(0, data.size());
            rand_index = distrib(gen);
            // rand_index = rand() % (data.size());
        } while (!IndexCheck(rand_index, indices));

        indices.push_back(rand_index);
    }
    PrintPoints(indices);
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
        return 0;
    }

    file_name = argv[1];
    num_cluster = stoi(argv[2]);
    max_iter = stoi(argv[3]);
    threshold = stod(argv[4]);
    num_runs = stoi(argv[5]);

    // Validate inputs
    if (num_cluster <= 1 || max_iter <= 0 || threshold < 0 || num_runs <= 0)
    {
        cout << "Invalid inputs" << endl;
        return 0;
    }

    // Read data and process clusters
    ReadFile(file_name);
    SelectCluster(num_cluster);

    return 0;
}