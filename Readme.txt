Submitted by : PESU_MI-0170_0225_1926
Implementation:
1. Cleaning the dataset:
1. Outliers were replaced with the median after being detected by the box plot method(IQR) only for BP column
2. Missing values were replaced with the median, mean wherever applicable. Mode was used for categorical columns.

   2. Neural network implementation: 

The neural network was implemented with three layers ( input, hidden and output). It has two functions: fit and predict. 
The fit function trains the dataset using forward and backward propagations iteratively to generate an accurate model. 
During forward propagation, the network uses tanh as its activation function and randomised weight and bias matrices to compute the output. The softmax function is used as the activation function for the output layer to obtain probabilities of the predicted classes. Batch gradient descent with a fixed learning rate is used to minimize the loss and maximise the likelihood by tweaking the weights and bias metrics using back-propagation to calculate the gradients with respect to the parameters. After each iteration a dictionary containing the parameters are updated. 
The fit function implements the above multiple times (epochs) to ensure the model gets trained and predicts accurately.
The predict function on the other hand uses the trained model to predict the output for the passed dataset. It implements forward propagation to predict the highest probability class, and returns the predicted values.
One-third of the dataset is used as test data. The model is trained by passing the training dataset to the fit function after which the test dataset is used to test the model using the predict function. 
The predicted values are then sent to the confusion matrix function along with the expected labels to calculate the accuracy and other metrics like F1 score, precision and recall. 
      3. Hyperparameters:
      1. The regularization strength is initialized to 1 (lambda)
      2. The learning rate is set as 0.0001 (epsilon)
      3. The training set size is the same as that of the size of the training data
      4. Activation functions used : tanh,softmax
      5. No of layers: 3
      6. No of hidden layers: 1
      7. No of neurons in the hidden layer : 3
      8. No of iterations :30,000
      9. Optimization algorithm : Gradient Descent
      10. Dimensionality of output layer : 2
      11. Dimensions of weight and bias matrices:
      1. Weight1: number of features x number of neurons in hidden layer
      2. Bias1: 1 x number of neurons in hidden layer
      3. Weight2: number of neurons in hidden layer x dimensionality of the output layer
      4. Bias2: 1 x  dimensionality of the output layer
      4. Key feature: Regularizing the weight gradients in each step to prevent overfitting.
      5. Something beyond basics: Regularization was implemented to generalize the model.
      6. Steps to run the file:
      1.  cd  src
      2. Install Libraries:
pip install pandas numpy sklearn
      3. Run the code (Neural network) :
python3 ANN.py
      4. To preprocess the data  :
cd data
python3 NN_cleaning.py


Comments:
The pre-processed csv file  (LBW_Dataset_Preprocessed.csv ) is there in both the src and data folder for easier access.
The original dataset is there in the src folder.