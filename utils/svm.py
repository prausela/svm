import numpy as np

class SVM:
    def __init__(self, c: float = 1.0, batch_size: int = 1, epochs: int = 1000, initial_learning_rate: float = 0.01, decay_rate: float = 0.0001):
        self.c = c
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.initial_learning_rate = initial_learning_rate
        self.curr_w = None
        self.curr_b = None
        self.best_w = None
        self.best_b = None
        self.best_cost = float("inf")

    def get_cost(self, x: np.ndarray, y: np.ndarray):
        reg = 0.5 * np.dot(self.curr_w, self.curr_w)
        sample_count = x.shape[0]
        loss = np.ndarray(sample_count)
        for i in range(sample_count):
            t = y[i] * (np.dot(self.curr_w, x[i]) + self.curr_b)
            loss[i] = reg + self.c * max(0, 1-t)
        return np.average(loss)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        sample_count, feature_count = x.shape

        self.curr_w = np.zeros(feature_count)
        self.curr_b = 0

        ids = np.arange(sample_count)
        np.random.shuffle(ids)

        costs = []

        for epoch in range(self.epochs):

            # exponential decay
            self.learning_rate = self.initial_learning_rate * np.exp(self.decay_rate * epoch * (-1))

            # mini batch gradient descent
            for batch_start in range(0, sample_count, self.batch_size):
                w_gradient = 0
                b_gradient = 0
                for j in range(batch_start, batch_start+self.batch_size):
                    if j < sample_count:
                        idx = ids[j]
                        t_i = y[idx] * (np.dot(self.curr_w, x[idx].T) + self.curr_b)
                        if t_i <= 1:
                            w_gradient -= self.c * y[idx] * x[idx]
                            b_gradient -= self.c * y[idx]
                self.curr_w = self.curr_w - self.learning_rate * w_gradient
                self.curr_b = self.curr_b - self.learning_rate * b_gradient

            # check the cost function
            curr_cost = self.get_cost(x, y)
            costs.append(curr_cost)

            # store the best result
            if curr_cost < self.best_cost:
                self.best_cost = curr_cost
                self.best_w = self.curr_w.copy()
                self.best_b = self.curr_b

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = np.dot(x, self.best_w) + self.best_b
        return np.sign(prediction)
