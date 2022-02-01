import numpy as np

class KNearstNeighbors:
    def __init__(self, k):
        self.k = k

    # train
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def euclidianDistance(self,a,b):
        d = np.sqrt(np.sum((a-b)**2))
        return d
    
    def nearNeighbors(self,X_test):
        dists = []
        for x_train in self.X_train:
            dist = self.euclidianDistance(x_train,X_test)
            dists.append(dist)
            
        index_sorted = np.argsort(dists)
        gender_sorted = self.Y_train[index_sorted]
        
        return gender_sorted[0:self.k]
    
    #test
    def predict(self,X_test):
        neighbors = self.nearNeighbors(X_test)
        Y_test = np.argmax(np.bincount(neighbors))
        
        return Y_test