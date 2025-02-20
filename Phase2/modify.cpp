#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

int main() {
    using std::ifstream;
    using std::ofstream;
    using std::vector;
    using std::string;
    using std::cout;
    using std::endl;
    
    ifstream input_file("iris_bezdek.txt");
    ofstream output_file("iris_bezdek_mod.txt");
    vector<string> dataset;
    string line;

    if (!input_file) 
    {
        cout << "Error opening input file!" << endl;
        return 1;
    }

    while (getline(input_file, line)) 
    {
        dataset.push_back(line);
    }
    input_file.close();
    
    for (int i = 0; i < dataset.size(); i++) 
    {
        output_file << dataset[i] << endl;
        if ((i + 1) % 5 == 0) 
        { 
            for (int j = 0; j < 9; j++) 
            {
                output_file << dataset[i] << endl;
            }
        }
    }

    output_file.close();

    return 0;
}
