import numpy as np

from utils.svm_utils import calculate_cost, compute_gradient, l

class SVM:
    def __init__(self, c: float = 1.0, batch_size: int = 1, iterations: int = 1000, initial_learning_rate: float = 0.01, decay_rate: float = 0.0001):
        self.c = c
        self.iterations = iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.initial_learning_rate = initial_learning_rate
        self.w = None
        self.b = None
        self.best_w = None
        self.best_b = None
        self.best_cost = 1e9

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        sample_count, feature_count = x.shape

        self.w = np.zeros(feature_count)
        self.b = 0

        # mini batch gradient descent
        for iteration in range(self.iterations):

            # exponential decay
            self.learning_rate = self.initial_learning_rate * np.exp(self.decay_rate * iteration * (-1))

            # get one random mini batch
            rand_idxs = np.random.choice(sample_count, self.batch_size, replace=False)
            x_batch = x[rand_idxs]
            y_batch = y[rand_idxs]

            w_gradient, b_gradient = compute_gradient(x_batch, y_batch, self.w, self.b, self.c)
            self.w -= self.learning_rate * w_gradient
            self.b -= self.learning_rate * b_gradient

            # check the cost function
            cost = calculate_cost(x_batch, y_batch, self.w, self.b, self.c)
            '''
            print(f"iteration: {iteration}; cost: {cost}; learning rate: {self.learning_rate}")
            '''

            # store the best result
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_w = self.w.copy()
                self.best_b = self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear_output = np.dot(x, self.best_w) + self.best_b
        return np.sign(linear_output)
