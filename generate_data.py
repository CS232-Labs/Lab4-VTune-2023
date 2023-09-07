import sys
import numpy as np
from tqdm import tqdm

def generate_data(n_samples, n_features, noise_stddev, bias, coefficients):
    """Generate training data based on linear function + gaussian noise."""
    X = np.random.randn(n_samples, n_features)
    noise = np.random.normal(0, noise_stddev, n_samples)
    
    Y = X.dot(coefficients) + bias + noise

    return X, Y

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_data.py <n_samples> <n_features>")
        sys.exit(1)

    n_samples = int(sys.argv[1])
    n_features = int(sys.argv[2])
    noise_stddev = 1
    bias = np.random.randn()

    # coefficients = [float(x) for x in input(f"Enter the coefficients for the input features, separated by spaces: ").split()]
    coefficients = np.random.randn(n_features)
    print(f'bias: {bias}, coefficients: {coefficients}')
    assert len(coefficients) == n_features, "Number of coefficients should match number of input features"

    X, Y = generate_data(n_samples, n_features, noise_stddev, bias, coefficients)

    with open("data.txt", "w") as file:
        file.write(f"{n_samples}\n{n_features}\n")
        for x, y in tqdm(zip(X, Y)):
            file.write(' '.join(map(str, x)) + ' ' + str(y) + '\n')

if __name__ == "__main__":
    main()
