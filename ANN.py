import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

df = pd.read_csv("LBW_Dataset_Preprocessed.csv")
X = df.iloc[:, :-1].values #selecting rows and columns of the dataframe
Y = df.iloc[:, -1].values #selecting rows and columns of dataframe
x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size=1/3 , random_state=2)#splitting into training and testing data

class NN:

    def __init__(self):
        self.train_size = len(x_train)  # size of training set
        self.nn_input_dim = x_train.shape[1] # input layer dimensions
        self.nn_output_dim = 2  # output layer dimensions

        # Gradient descent parameters 
        self.epsilon = 0.0001 # learning rate for gradient descent
        self.reg_lambda = 1 # regularization strength
    

    def predict(self,model, x):
        self.W1, self.w1, self.W2, self.w2 = model['W1'], model['w1'], model['W2'], model['w2']
        # Forward propagation
        self.m1 = np.dot(x,self.W1) + self.w1 #m1=W1.x + w1
        self.n1 = np.tanh(self.m1) #n1=tanh(m1)
        self.m2 = self.n1.dot(self.W2) + self.w2 #m2=n1.W2 + w2
        self.exp_score = np.exp(self.m2) #exponential of m2
        self.probs = self.exp_score / np.sum(self.exp_score, axis=1, keepdims=True)
        return np.argmax(self.probs, axis=1)


    def fit(self,x_train,y,nn_hidden_dim=3, num_passes=30000, print_loss=False):
        np.random.seed(42)# Initializing parameters to random values.

        self.W1 = np.random.randn(self.nn_input_dim, nn_hidden_dim) *30
        self.w1 = np.zeros((1, nn_hidden_dim))

        self.W2 = np.random.randn(nn_hidden_dim, self.nn_output_dim) *40
        self.w2 = np.zeros((1, self.nn_output_dim))

        model = {}
        
        # Gradient descent.For each batch-
        for i in range(0, num_passes):

            # Forward propagation
            self.m1 = np.dot(x_train,self.W1) + self.w1 #m1=W1.x + w1
            self.n1 = np.tanh(self.m1) #n1=tanh(m1)
            self.m2 = self.n1.dot(self.W2) + self.w2 #m2=n1.W2 + w2
            self.exp_score = np.exp(self.m2) #exponential of m2
            self.probs = self.exp_score / np.sum(self.exp_score, axis=1, keepdims=True)

            # Backpropagation
            self.d3 = self.probs
            
            self.d3[range(self.train_size), y] -= 1
            self.dW2 = (self.n1.T).dot(self.d3)
            self.dw2 = np.sum(self.d3, axis=0, keepdims=True)
            self.d2 = self.d3.dot(self.W2.T) * (1 - np.power(self.n1, 2))
            self.dW1 = np.dot(x_train.T, self.d2)
            self.dw1 = np.sum(self.d2, axis=0)

            # Add regularization terms 
            self.dW2 += self.reg_lambda * self.W2
            self.dW1 += self.reg_lambda * self.W1
            
            # Gradient descent parameter update
            self.W1 += -self.epsilon * self.dW1
            self.w1 += -self.epsilon * self.dw1
            self.W2 += -self.epsilon * self.dW2
            self.w2 += -self.epsilon * self.dw2
            
            # Assign new parameters to the model
            model = { 'W1': self.W1, 'w1': self.w1, 'W2': self.W2, 'w2': self.w2}
        return model


    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        self.cm=[[0,0],[0,0]]
        self.fp=0
        self.fn=0
        self.tp=0
        self.tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                self.tp=self.tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                self.tn=self.tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                self.fp=self.fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                self.fn=self.fn+1
            
        self.cm[0][0]=self.tn
        self.cm[0][1]=self.fp
        self.cm[1][0]=self.fn
        self.cm[1][1]=self.tp
            
        self.p= self.tp/(self.tp+self.fp)
        self.r=self.tp/(self.tp+self.fn)
        self.f1=(2*self.p*self.r)/(self.p+self.r)
        self.acc = (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        
        print("Confusion Matrix : ")
        print(self.cm)
        print("\n")
        print(f"Precision : {self.p}")
        print(f"Recall : {self.r}")
        print(f"F1 SCORE : {self.f1}")
        print(f"Accuracy : {self.acc}")
            
neural_network = NN()

model = neural_network.fit(x_train=x_train,y=y_train)  # train the network
nn_p = neural_network.predict(model, x_test).reshape(x_test.shape[0],).tolist()  #predicted list
neural_network.CM(y_test, nn_p) # Metrics