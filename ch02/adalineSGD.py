import numpy as np

class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, X, y):
        self.losses_ = []
        self.intialize_weights(X.shape[1])
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for x, target in zip(X,y):
                losses.append(self._update_weights(x, target))
            self.losses_.append(np.mean(losses))
        return self


    def _update_weights(self, xi, target):
        output = np.dot(xi, self.w_) + self.b_
        error = target - output
        self.b_ += error *2.0 * self.eta
        self.w_ += 2.0 * self.eta * error * xi
        loss = error**2
        return loss

    def intialize_weights(self, m):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
    
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def predict(self, X):
        return np.where(np.dot(X, self.w_) + self.b_ >= 0.5, 1, 0)
