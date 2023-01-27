class Adaline:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses = []

        # In Adaline we have to update gradient for all examples at once
        for _ in range(self.n_iter):
            error = (y - self.predict(X))
            self.b_ += self.eta * 2.0 * np.mean(error)
            update = (self.eta * 2.0 * np.dot(np.transpose((error)), np.transpose(X))) / X.shape[0]
            loss = np.mean(error**2)
            self.losses.append(loss)
        return self
   
    
    def predict(self, x):
        return np.dot(x, self.w_) + self.b_