import numpy as np
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_regression
import time
from scipy.special import digamma


class JoinAwareMIEstimator:
    """
    Initialize MI estimator with k nearest neighbors

    Args:
        k: Number of nearest neighbors to use
    """
    def __init__(self, k=3):
        self.k = k
        self._valid_keys = None
        self._x_values = None
        self._y_values = None

    def _compute_distances(self, X, Y, join_key):
        """
        Compute k-nearest neighbor distances without materializing full join
        
        Args:
            X: Dictionary mapping join_key -> x_value 
            Y: Dictionary mapping join_key -> y_value
            join_key: Array of join keys that exist in both X and Y
        
        Returns:
            epsilon: Array of k-th nearest neighbor distances
        """
        x_values = np.array([X[k] for k in join_key])
        y_values = np.array([Y[k] for k in join_key])
        n_samples = len(join_key)

        distances = []
        for i in range(n_samples):
            # Compute distances in both dimensions using maximum norm
            dx = np.abs(x_values - x_values[i])
            dy = np.abs(y_values - y_values[i])
            dist = np.maximum(dx, dy)  # L-infinity norm

            # Sort distances and get k-th nearest neighbor distance
            # Skip the point itself (hence k+1)
            dist.sort()
            if len(dist) > self.k:
                distances.append(dist[self.k])

        return np.array(distances)

    def _count_neighbors(self, X, Y, join_key, epsilon):
        """
        Count points within epsilon radius for each dimension
        
        Args:
            X, Y: Dictionaries mapping join_key -> value
            join_key: Array of valid join keys
            epsilon: Array of distances for each point
            
        Returns:
            nx, ny: Arrays containing number of neighbors in each dimension
        """
        x_values = np.array([X[k] for k in join_key])
        y_values = np.array([Y[k] for k in join_key])
        n_samples = len(join_key)

        nx = np.zeros(len(epsilon), dtype=np.int32)
        ny = np.zeros(len(epsilon), dtype=np.int32)

        idx = 0
        for i in range(n_samples):
            # Skip points that weren't included in epsilon calculation
            if idx >= len(epsilon):
                break

            eps = epsilon[idx]
            # Count points within epsilon in each dimension separately
            nx[idx] = np.sum(np.abs(x_values - x_values[i]) <= eps)
            ny[idx] = np.sum(np.abs(y_values - y_values[i]) <= eps)
            idx += 1

        return nx, ny

    def estimate(self, X, Y, join_key=None):
        """
        Estimate mutual information between X and Y using k-nearest neighbors
        
        Args:
            X: Dictionary mapping join_key -> x_value
            Y: Dictionary mapping join_key -> y_value 
            join_key: Optional array of join keys present in both X and Y. If None, will compute intersection of X and Y keys.
            
        Returns:
            mi: Estimated mutual information value
        """
        if join_key is None:
            join_key = list(set(X.keys()) & set(Y.keys()))

        if len(join_key) < self.k + 1:  # Need at least k+1 points
            return 0.0

        # Get k-th nearest neighbor distances
        epsilon = self._compute_distances(X, Y, join_key)
        if len(epsilon) == 0:
            return 0.0

        # Count neighbors within epsilon radius
        nx, ny = self._count_neighbors(X, Y, join_key, epsilon)

        # Apply estimator from equation (9) in the paper
        n = len(epsilon)
        mi = digamma(self.k) - 1/self.k - \
            np.mean(digamma(nx) + digamma(ny)) + digamma(n)

        return max(0, mi)


def generate_test_data(n_samples=1000, correlation=0.5, noise=0.1, missing_frac=0.1):
    """Generate correlated normal distributions with known MI"""
    # Generate correlated normal data
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    x, y = np.random.multivariate_normal(mean, cov, n_samples).T
    
    x += np.random.normal(0, noise, n_samples)
    y += np.random.normal(0, noise, n_samples)

    all_keys = [f'key{i}' for i in range(n_samples)]
    missing_keys = np.random.choice(all_keys,size=int(missing_frac * n_samples),replace=False)
    X = {f'key{i}': val for i, val in enumerate(x) if f'key{i}' not in missing_keys}
    Y = {f'key{i}': val for i, val in enumerate(y) if f'key{i}' not in missing_keys}

    # Theoretical MI for bivariate normal is -0.5 * log(1 - correlation^2)
    theoretical_mi = -0.5 * np.log(1 - correlation**2)

    return X, Y, theoretical_mi


def test_mi_estimators(n_samples=1000, correlations=[0.0, 0.3, 0.6, 0.9], k=3):
    print("\nTesting MI Estimators")
    print("-" * 80)
    print(f"{'Correlation':^12} | {'Theoretical':^12} | {'Sklearn':^12} | {'JoinAware':^12} | {'Time Ratio':^12}")
    print("-" * 80)

    for correlation in correlations:
        # Generate test data
        X_dict, Y_dict, theoretical_mi = generate_test_data(n_samples, correlation)
        join_keys = list(X_dict.keys())

        # Convert to array format for sklearn
        X_arr = np.array([X_dict[k] for k in join_keys]).reshape(-1, 1)
        y_arr = np.array([Y_dict[k] for k in join_keys])

        # Time sklearn implementation
        t0 = time.time()
        sklearn_mi = mutual_info_regression(X_arr, y_arr, n_neighbors=k)[0]
        sklearn_time = time.time() - t0

        # Time our implementation
        t0 = time.time()
        estimator = JoinAwareMIEstimator(k=k)
        our_mi = estimator.estimate(X_dict, Y_dict, join_keys)
        our_time = time.time() - t0

        # Print results
        print(f"{correlation:^12.3f} | {theoretical_mi:^12.3f} | {sklearn_mi:^12.3f} | {our_mi:^12.3f} | {our_time/sklearn_time:^12.3f}")


if __name__ == "__main__":
    np.random.seed(42)

    # Test with different sample sizes
    for n_samples in [100, 1000, 10000]:
        print(f"\nTesting with {n_samples} samples:")
        test_mi_estimators(n_samples=n_samples)

    # Test with different k values
    print("\nTesting with different k values (1000 samples):")
    for k in [1, 3, 5, 10]:
        print(f"\nk = {k}:")
        test_mi_estimators(n_samples=1000, k=k)
