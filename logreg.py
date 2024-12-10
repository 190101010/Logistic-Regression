import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def computeCost(self, theta, X, y, regLambda):
        m = y.size
        h = self.sigmoid(X @ theta)

        # Cost without regularization
        cost = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))

        # Regularization term (exclude theta[0] from regularization)
        reg_term = (regLambda / (2 * m)) * np.sum(np.square(theta[1:]))

        return cost + reg_term



    def computeGradient(self, theta, X, y, regLambda):
        m = y.size
        h = self.sigmoid(X @ theta)
        error = h - y

        # Gradient calculation
        grad = (1 / m) * (X.T @ error)

        # Regularization (excluding theta[0])
        reg_term = (regLambda / m) * np.r_[[0], theta[1:]]

        return grad + reg_term


    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            theta_new = self.theta - self.alpha * gradient

            # Check for convergence
            if np.linalg.norm(theta_new - self.theta) < self.epsilon:
                break

            self.theta = theta_new


    def predict(self, X):
        return self.sigmoid(X @ self.theta) >= 0.5
