#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Vector>

using namespace std;
using namespace Eigen;

//g++ -I C:\Users\saaja\Desktop\College\UMAP umap.cpp -o umap

/* template <typename Derived>
MatrixXd euclidean_distance(const MatrixBase<Derived>& data){*/


MatrixXd load_data(string filename) {
    vector<double> matrixVec;
    string filepath = filename, row, element;
    int rowNum = 0;
    ifstream dataset(filepath);
    
    getline(dataset, row);
    while (getline(dataset, row)){
        stringstream rowString(row);
        int i = 0;
        while(getline(rowString, element, '\t')){
            if(i != 0){
                matrixVec.push_back(stod(element));
            }
            i++;
        }
        rowNum++;
    }
    MatrixXd data = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrixVec.data(), rowNum, matrixVec.size() / rowNum);
    data.conservativeResize(data.rows(), data.cols() - 1);
    //data.conservativeResize(data.rows(), data.cols());
    return data;
}


MatrixXd euclidean_distance(MatrixXd &data){
    MatrixXd distances, temp;
    double dist;
    vector<double> distsVec;
    vector<VectorXd> v;
    for(int i = 0; i < data.rows(); i++){ 
        v.push_back(data.row(i));
        temp = (-data).rowwise() + v[0].transpose();
        temp = temp.array().pow(2).matrix();
        for(int j = 0; j < temp.rows(); j++){
            distsVec.push_back(sqrt(temp.row(j).sum()));
        }
        v.clear();
    }
    distances = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(distsVec.data(), data.rows(), data.rows());
    distances.conservativeResize(data.rows(), data.rows());
    return distances;
}


MatrixXd Total_probability_matrix(MatrixXd &distances, vector<double> &rho, vector<double> &sigma){
    MatrixXd prob = distances;
    for(int i = 0; i < prob.rows(); i++){
        prob.row(i) = prob.row(i).array() - rho[i];
        for(int j = 0; j < prob.cols(); j++){
            if(prob(i, j) < 0){
                prob(i, j) = 0;
            }
        }
        prob.row(i) = (prob.row(i).array() / sigma[i]).exp();
    }
    return prob;
}

double probability_rowwise(MatrixXd &distances, int rownum, double rho, double sigma){
    vector<VectorXd> v;
    double k;
    v.push_back(distances.row(rownum));
    v[0] = ((v[0].array() - rho) / sigma).exp();
    k = v[0].sum();
    return(pow(2, k));
}


vector<double> sigma(double k, int iter, MatrixXd &distances, vector<double> rho){
    vector<double> sigmaValues;
    for(int i = 0; i < distances.rows(); i++){
        double lower = 0, upper = 1000, temp = 0;
        for(int j = 0; j < 20; j++){
            temp = (upper + lower) / 2;
            double sigma_k = probability_rowwise(distances, i, rho[i], temp);
            if(sigma_k < k){
                lower = temp;
            }
            else{
                upper = temp;
            }
            if(abs(k - sigma_k) < 0.001){
                break;
            }
        }
        sigmaValues.push_back(temp);
    }
    return sigmaValues;
}


int main() {
    MatrixXd dataset = load_data("./data.txt");
    //cout.precision(17);
    dataset = (dataset.array() + 1).matrix();
    dataset = dataset.array().log().matrix();
    cout << dataset.rows() << " rows and " << dataset.cols() << " columns.";
    //cout << endl << dataset << endl << endl;

    MatrixXd distances;
    double dist;
    distances = euclidean_distance(dataset);
    //cout << distances.rows() << endl << distances.cols();
    //cout << distances << endl;

    vector<double> rho, v;
    for(int i = 0; i < distances.rows(); i++){
        for(int j = 0; j < distances.cols(); j++){
            v.push_back(distances(i, j));
        }
        sort(begin(v), end(v));
        rho.push_back(v[1]);
        v.clear();
    }

    cout << "\n\nRho values:  ";
    cout << rho.size() << "\n\n";
    /* for(auto i:rho){
        cout << i << "  ";
    }
    cout << "\n\n"; */
    
    vector<double> sigmaVals = sigma(15, 20, dataset, rho);
    cout << "Sigma values:  ";
    cout << sigmaVals.size() << "\n\n";
    /* 
    for(auto i:sigmaVals){
        cout << i << "  ";
    }
    cout << "\n\n"; */

    MatrixXd probability_matrix;
    probability_matrix = Total_probability_matrix(distances, rho, sigmaVals);
    cout << "Probability matrix:\n";
    cout << probability_matrix.rows() << " rows and " << probability_matrix.cols() << "columns.\n";
    
    return 0;
}