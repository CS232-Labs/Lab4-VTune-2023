#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <chrono>

class LinearRegression {
private:
    std::vector<double> weights;
    std::vector<std::vector<double> > X;
    std::vector<double> Y;

    static std::vector<double> matrix_vector_multiply(std::vector<std::vector<double> > matr, std::vector<double> vec)
    {
        size_t n = matr.size();
        assert(n > 0);
        size_t d = matr[0].size();
        assert(d > 0);
        for(size_t i = 1; i < n; ++i) assert(matr[i].size() == d);
        assert(vec.size() == d);
        std::vector<double> ret_val;
        for(size_t i = 0; i < n; ++i)
        {
            double sum = 0;
            for(size_t j = 0; j < d; ++j)
            {
                sum += matr[i][j]*vec[j];
            }
            ret_val.push_back(sum);
        }
        return ret_val;
    }

    static std::vector<std::vector<double> > matrix_transpose(std::vector<std::vector<double> > matr)
    {
        size_t n = matr.size();
        assert(n > 0);
        size_t d = matr[0].size();
        assert(d > 0);
        for(size_t i = 1; i < n; ++i) assert(matr[i].size() == d);
        std::vector<std::vector<double> > ret_val;
        for(size_t j = 0; j < d; ++j)
        {
            std::vector<double> v;
            for(size_t i = 0; i < n; ++i)
            {
                v.push_back(matr[i][j]);
            }
            ret_val.push_back(v);
        }
        return ret_val;
    }

    static std::vector<std::vector<double> > matrix_multiply(std::vector<std::vector<double> > A, std::vector<std::vector<double> > B)
    {
        size_t n = A.size();
        assert(n > 0);
        size_t k = A[0].size();
        assert(k > 0);
        for(size_t i = 1; i < n; ++i) assert(A[i].size() == k);
        assert(B.size() == k);
        size_t d = B[0].size();
        assert(d > 0);
        for(size_t i = 1; i < k; ++i) assert(B[i].size() == d);
        std::vector<std::vector<double> > ret_val;
        for(size_t i = 0; i < n; ++i)
        {
            std::vector<double> v;
            for(size_t j = 0; j < d; ++j)
            {
                double sum = 0;
                for(size_t p = 0; p < k; ++p)
                {
                    std::vector<double> a_i = A[i];
                    double a_ip = a_i[p];
                    std::vector<double> b_p = B[p];
                    double b_pj = b_p[j];
                    sum += a_ip*b_pj;
                }
                v.push_back(sum);
            }
            ret_val.push_back(v);
        }
        return ret_val;
    }

    static std::vector<std::vector<double> > matrix_inverse(std::vector<std::vector<double> > matr)
    {
        size_t n = matr.size();
        assert(n > 0);
        for(size_t i = 0; i < n; ++i) assert(matr[i].size() == n);
        std::vector<std::vector<double>> matrix = matr;

        // Augment the matrix [A | I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j)
                    matrix[i].push_back(1);
                else
                    matrix[i].push_back(0);
            }
        }

        for (int i = 0; i < n; i++) {
            assert(std::abs(matrix[i][i]) > 1e-10);

            // Make the diagonal contain all ones
            double diag = matrix[i][i];
            for (int j = 0; j < 2*n; j++) {
                matrix[i][j] /= diag;
            }

            // Make the other rows contain zeros
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double ratio = matrix[j][i];
                    for (int k = 0; k < 2*n; k++) {
                        matrix[j][k] -= ratio * matrix[i][k];
                    }
                }
            }
        }

        // Extract the inverse matrix from [I | A^-1]
        std::vector<std::vector<double>> inverse;
        for (int i = 0; i < n; i++) {
            std::vector<double> row;
            for (int j = n; j < 2*n; j++) {
                row.push_back(matrix[i][j]);
            }
            inverse.push_back(row);
        }

        return inverse;
    }

public:
    LinearRegression(std::vector<std::vector<double>> x, std::vector<double> y)
    {
        assert(x.size() == y.size());
        size_t n = x.size();
        assert(n > 0);
        size_t d = x[0].size();
        assert(d > 0);
        for(size_t i = 1; i < n; ++i) assert(x[i].size() == d);
        X.resize(n, {1});
        weights.resize(d + 1, 0);
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < d; ++j) X[i].push_back(x[i][j]);
        }
        Y = y;
    }

    void fit() 
    {
        weights = matrix_vector_multiply(matrix_multiply(matrix_inverse(matrix_multiply(matrix_transpose(X), X)), matrix_transpose(X)), Y);
    }

    double loss() const
    {
        double sum = 0;
        for(size_t i = 0; i < Y.size(); ++i)
        {
            double err = (Y[i] - predict(X[i]));
            sum += err*err;
        }
        return sum/Y.size();
    }

    double predict(std::vector<double> x) const 
    {
        double sum = 0;
        for (size_t i = 0; i < x.size(); ++i) 
        {
            sum += weights[i] * x[i];
        }
        return sum;
    }
};

int main(int argc, char **argv) {
    // Open data.txt for reading
    std::ifstream infile("data.txt");
    if (!infile.is_open()) {
        std::cerr << "Error opening data.txt" << std::endl;
        return 1;
    }

    // Read in the number of training examples
    int n_samples;
    infile >> n_samples;

    // Read in the number of input features
    int n_features;
    infile >> n_features;

    // Read in the training examples
    std::vector<std::vector<double>> X(n_samples, std::vector<double>(n_features));
    std::vector<double> y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            infile >> X[i][j];
        }
        infile >> y[i];
    }

    infile.close(); // Close the file

    auto start = std::chrono::high_resolution_clock::now();

    LinearRegression model(X, y);
    model.fit();
    double loss = model.loss();

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Training Loss: " << loss << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1e6 << " seconds" << std::endl;

    return 0;
}
