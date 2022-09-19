#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Vector>

using namespace std;
using namespace Eigen;
double a = 1.121436342369708, b = 1.057499876613678;


MatrixXd load_data(string filename)
{
    vector<double> matrixVec;
    string filepath = filename, row, element;
    int rowNum = 0;
    ifstream dataset(filepath);

    getline(dataset, row);
    while (getline(dataset, row))
    {
        stringstream rowString(row);
        int i = 0;
        while (getline(rowString, element, '\t'))
        {
            if (i != 0)
            {
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

//MatrixXd euclidean_distance(MatrixXd &data){
//MatrixXd result(data.rows(),data.rows());
//	MatrixXd distances, temp;
//    for(int i = 0; i < data.rows(); i++){
//        temp = (-data).rowwise() + data.row(i);
//        result.row(i)=temp.rowwise().squaredNorm();
//        cout<<i<<endl;
//    }
//    return result;
//}

MatrixXd euclidean_distance(MatrixXd &data)
{
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic> XX, XY, result;
    int N = data.rows();
    XX.resize(N, 1);
    XY.resize(N, N);
    result.resize(N, N);
    XX = data.rowwise().squaredNorm();
    XY = 2 * data * data.transpose();
    result = XX * MatrixXd::Ones(1, N);
    result = result + MatrixXd::Ones(N, 1) * XX.transpose();
    result = result - XY;
    return result;
}

MatrixXd Total_probability_matrix(MatrixXd &distances, vector<double> &rho, vector<double> &sigma)
{
	fstream myfile;
    myfile.open("check1.txt",fstream::out);

    MatrixXd prob = distances;
    for (int i = 0; i < prob.rows(); i++)
    {
        prob.row(i) = prob.row(i).array() - rho[i];
        for (int j = 0; j < prob.cols(); j++)
        {
            if (prob(i, j) < 0)
            {
                prob(i, j) = 0;
            }
        }
        // prob.row(i) = ((prob.row(i).array()*-1) / sigma[i]).exp();
        myfile << ((prob.row(i).array()*-1) / sigma[i]).exp() << endl;
    }
	myfile.close();
    
    return prob;
}

double probability_rowwise(MatrixXd &distances, int rownum, double rho, double sigma)
{
    vector<VectorXd> v;
    double k = 0;
    v.push_back(distances.row(rownum));

    //	v[0].array()[rownum]=0;
    //    v[0] = ((-1*(v[0].array()-rho))/ sigma).exp();
    for (int i = 0; i < v[0].rows(); i++)
    {
        double temp = v[0].row(i).array()[0];
        if (temp - rho < 0)
            k += 1;
        else
            k += exp((rho - temp) / sigma);
    }

    //    cout<<"K: "<<k<<endl;
    //	cout<<"POW: "<<pow(2, k)<<endl;
    return (pow(2, k));
}

vector<double> sigma(double k, int iter, MatrixXd &distances, vector<double> rho)
{
    vector<double> sigmaValues;
    for (int i = 0; i < distances.rows(); i++)
    {
        double lower = 0, upper = 1000, temp = 0;
        for (int j = 0; j < iter; j++)
        {
            temp = (upper + lower) / 2.0;
            double sigma_k = probability_rowwise(distances, i, rho[i], temp);
            if (sigma_k < k)
            {
//                cout << "Here";
//                cout << temp << endl;
                lower = temp;
            }
            else
            {
                upper = temp;
            }
            if (abs(k - sigma_k) < 0.00001)
            {
                break;
            }
//            cout << "Sigma K: " << sigma_k << endl;
//            cout << "Approx Sigma : " << temp << endl;
        }

        //        if ((i + 1) % 100 == 0)
        //        		cout<<"\nSigma binary search finished "<< i+ 1<<"of "<<distances.rows()<<" cells";
        sigmaValues.push_back(temp);
    }
    return sigmaValues;
}

MatrixXd low_dim_prob(MatrixXd &data)
{
    MatrixXd euclid = euclidean_distance(data);
    return (((euclid.array().pow(b)) * a) + 1).pow(-1);
}
int main()
{
    cout.precision(17);
    MatrixXd dataset = load_data("./data.txt");
    //cout.precision(17);
    dataset = (dataset.array() + 1).matrix();
    dataset = dataset.array().log().matrix();
    cout << dataset.rows() << " rows and " << dataset.cols() << " columns.";
    //cout << endl << dataset << endl << endl;

    MatrixXd distances;
    double dist;
    distances = euclidean_distance(dataset);
    cout << distances.rows() << endl
         << distances.cols();

    vector<double> rho, v;
    for (int i = 0; i < distances.rows(); i++)
    {
        for (int j = 0; j < distances.cols(); j++)
        {
            v.push_back(distances(i, j));
        }
        sort(begin(v), end(v));
        rho.push_back(v[1]);
        v.clear();
    }

    //    cout << "\n\nRho values:  ";
    //    cout << rho.size() << "\n\n";
    //    for(auto i:rho){
    //        cout << i << "  ";
    //    }
    //    cout << "\n\n";

    vector<double> sigmaVals = sigma(15, 20, distances, rho);
    cout << "Sigma values:  ";
    cout << sigmaVals.size() << "\n\n";

//    for (auto i : sigmaVals)
//    {
//        cout << i << "  ";
//    }
//    cout << "\n\n";

    MatrixXd probability_matrix;
    
    probability_matrix = Total_probability_matrix(distances, rho, sigmaVals);
    // cout << "Probability matrix:\n";
    // cout << probability_matrix.rows() << " rows and " << probability_matrix.cols() << "columns.\n";
    // cout<< probability_matrix.row(probability_matrix.rows()-1);	
	return 0;
}
