import numpy as np

class NaiveBayes:

    def fit(self, X,y):
        nSamples, nFeatures = X.shape
        self._classes = np.unique(y)
        nClasses= len(self._classes)

        self._mean = np.zeros((nClasses, nFeatures) , dtype=np.float64)
        self._var = np.zeros((nClasses, nFeatures) , dtype=np.float64)
        self._priors = np.zeros(nClasses , dtype=np.float64)

        for idx , c in enumerate(self._classes):
            Xc = X[y == c]
            self._mean[idx, :] = Xc.mean(axis=0)
            self._var[idx, :] = Xc.var(axis=0)
            self._priors[idx] = Xc.shape[0] / float(nSamples)


    def predict(self, X):
        yPred = [self._predict(x) for x in X]
        return np.array(yPred)
    
    def _predict(self , x):
        posteriors = []

        for idx , c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx , x )))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
        
    def _pdf(self , classIdx , x):
        mean = self._mean[classIdx]
        var = self._var[classIdx]

        numr = np.exp(-((x - mean)**2 )/ (2* var))
        denm = np.sqrt(2* np.pi * var)
        return numr/denm

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))