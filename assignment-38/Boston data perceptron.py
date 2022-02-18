import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
import warnings
from numpy.linalg import inv
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    boston_dataset = load_boston()


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

X = boston[['LSTAT', 'RM']]
y = boston_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

class LinearListSquare:
    def __init__(self):
        pass

    def fit(self,X_train,Y_train):
        self.w = np.matmul(inv(np.matmul(X_train.T,X_train)), np.matmul(X_train.T,Y_train))
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        
        N = self.X_train.shape[0]
        self.learning_rate = 0.05
        self.epochs = 2

        # init weights
        self.W = np.random.rand(2, 1)
        fig = plt.figure(figsize=(12, 6))
        
        self.Errors = []

        x_range = np.arange(self.X_train[:,0].min(), self.X_train[:,0].max())
        y_range = np.arange(self.X_train[:,1].min(), self.X_train[:,1].max())
        
        # Train
        for self.epochs in range(self.epochs):
            for i in range(N):
                x = self.X_train[i,:]
                y_pred = np.matmul(x, self.W)
                e = self.Y_train[i] - y_pred
                # update weights
                x = x.reshape(-1, 1)
                self.W += self.learning_rate * e * x

                # visualization
                fig.clear()
                Y_pred = np.matmul(self.X_train, self.W)
                ax = fig.add_subplot(121,projection='3d')
                ax.clear()
                ax.scatter(self.X_train[:, 0], self.X_train[:, 1], self.Y_train, c='#0000ff')

                x, y = np.meshgrid(x_range, y_range)
                z = self.W[0] * x + self.W[1] * y
                ax.plot_surface(x, y, z, alpha=0.4)
                ax.set_xlabel("CRIM")
                ax.set_ylabel("TAX")
                ax.set_zlabel("MEDV")

                Error = np.mean(np.abs(self.Y_train - Y_pred))
                self.Errors.append(Error)

                ax2 = fig.add_subplot(122)
                ax2.clear()
                ax2.plot(np.arange(0,i+1), self.Errors)
                ax2.set_xlabel("Iteration #")
                ax2.set_ylabel("Cost")
                ax2.set_title('Training Curve')
                
                plt.pause(0.01)
        plt.show()


    # def predict(self, X_test):
        y_pred= np.matmul(X_test,self.w)
        return y_pred                     
    
    def evaluate(self , X , Y , loss="MAE"):
        Y_pred = []
        for i in range(X.shape[0]):
            y_pred = self.predict(X[i])
            Y_pred.append(y_pred)
            
        Y_pred = np.array(Y_pred)
        Error = Y - Y_pred
        
        if loss == "MAE":
            return np.mean(np.abs(Error))
        elif loss == "MSE":
            return np.mean(Error**2)


lls = LinearListSquare()
lls.fit(X,y)

y_pred = lls.predict(X_test)

MAE = lls.evaluate(X_test, y_test,'MAE')
MSE = lls.evaluate(X_test, y_test,'MSE')

print('MAE = ',MAE)
print('MSE = ' ,MSE)