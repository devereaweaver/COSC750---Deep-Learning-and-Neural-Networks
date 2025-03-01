# import numpy as np


# class NeuralNetwork(object):
#     def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
#         # Set number of nodes in input, hidden and output layers.
#         self.input_nodes = input_nodes
#         self.hidden_nodes = hidden_nodes
#         self.output_nodes = output_nodes

#         # Initialize weights
#         self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
#                                        (self.input_nodes, self.hidden_nodes))

#         self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
#                                        (self.hidden_nodes, self.output_nodes))
#         self.lr = learning_rate
        
#         #### TODO: Set self.activation_function to your implemented sigmoid function ####
#         #
#         # Note: in Python, you can define a function with a lambda expression,
#         # as shown below.
#         self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        
#         ### If the lambda code above is not something you're familiar with,
#         # You can uncomment out the following three lines and put your 
#         # implementation there instead.
#         #
#         #def sigmoid(x):
#         #    return 0  # Replace 0 with your sigmoid calculation here
#         #self.activation_function = sigmoid
                    

#     def train(self, features, targets):
#         ''' Train the network on batch of features and targets. 
        
#             Arguments
#             ---------
            
#             features: 2D array, each row is one data record, each column is a feature
#             targets: 1D array of target values
        
#         '''
#         n_records = features.shape[0]
#         delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
#         delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
#         for X, y in zip(features, targets):
            
#             final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
#             # Implement the backproagation function below
#             delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
#                                                                         delta_weights_i_h, delta_weights_h_o)
#         self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


#     def forward_pass_train(self, X):
#         ''' Implement forward pass here 
         
#             Arguments
#             ---------
#             X: features batch

#         '''
#         #### Implement the forward pass here ####
#         ### Forward pass ###
#         # TODO: Hidden layer - Replace these values with your calculations.
#         hidden_inputs = None # signals into hidden layer
#         hidden_outputs_before_activation = None # signals into the activation function of the hidden layer
#         hidden_outputs = None # signals from hidden layer

#         # TODO: Output layer - Replace these values with your calculations.
#         final_inputs = None # signals into final output layer
#         final_outputs = None # signals from final output layer
        
#         return final_outputs, hidden_outputs

#     def backpropagation(self, final_outputs, hidden_outputs, X, t, delta_weights_i_h, delta_weights_h_o):
#         ''' Implement backpropagation / forward pass here
         
#             Arguments
#             ---------
#             final_outputs: output from forward pass
#             t: target (i.e. label) batch
#             delta_weights_i_h: change in weights from hidden inputs to hidden outputs
#             delta_weights_h_o: change in weights from final inputs to final outputs

#         '''
#         #### Implement the backward pass here ####
#         ### Backward pass ###


#         # TODO: Calculate the output of the output layer contribution to error
#         # final_outputs_error = dL / dy
#         final_outputs_error = None # final outputs error is the difference between desired target and actual output.

#         # TODO: Calculate the output of the hidden layer's contribution to the error
#         # hidden_outputs_error = dL / dh
#         hidden_outputs_error = None  # Note the transpose here

#         # TODO: Calculate hidden_outputs_before_activation contribution to the error
#         # hidden_outputs_before_activation_error = dL / dh*
#         hidden_outputs_before_activation_error = None

#         # TODO: Weight step (hidden to output)
#         # delta_weights_h_o = dL / dW_2
#         delta_weights_h_o += None

#         # TODO: Weight step (input to hidden)
#         # delta_weights_i_h = dL / dW_1
#         delta_weights_i_h += None

#         return delta_weights_i_h, delta_weights_h_o

#     def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
#         ''' Update weights on gradient descent step
         
#             Arguments
#             ---------
#             delta_weights_i_h: change in weights from hidden inputs to hidden outputs
#             delta_weights_h_o: change in weights from final inputs to final outputs
#             n_records: number of records

#         '''
        
#         # Update the weights with gradient descent step
#         self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
#         self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

#     def run(self, features):
#         ''' Run a forward pass through the network with input features 
#             This function would be called by unit test functions, not for training purposes.
#             Arguments
#             ---------
#             features: 1D array of feature values
#         '''
        
#         #### Implement the forward pass here ####
#         # TODO: Hidden layer - Replace these values with your calculations.
#         hidden_inputs = None # signals into hidden layer
#         hidden_outputs_before_activation = None # signals into the activation function of the hidden layer
#         hidden_outputs = None # signals from hidden layer

#         # TODO: Output layer - Replace these values with your calculations.
#         final_inputs = None # signals into final output layer
#         final_outputs = None # signals from final output layer
        
#         return final_outputs


# #########################################################
# # Set your hyperparameters here
# ##########################################################
# iterations = 100
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1

import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes


        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))
        self.weights_input_to_hidden = np.array(self.weights_input_to_hidden, dtype=np.float64)

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))
        self.weights_hidden_to_output = np.array(self.weights_hidden_to_output, dtype=np.float64)
        self.lr = learning_rate
        self.lr = np.float64(self.lr)

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        #self.activation_function = lambda x: 1 / (1 + np.exp(-x.astype(np.float64)))

    def forward_pass_train(self, X):
        """Implement forward pass"""
        X = np.array(X, dtype=np.float64)
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # Signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # Signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # Signals into final output layer
        final_outputs = final_inputs  # No activation function for output layer (regression task)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        """Implement backpropagation"""
        error = y - final_outputs  # Output layer error
        output_error_term = error  # Derivative of f(x) = x is 1

        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)  # Backpropagated error
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)  # Backprop through sigmoid

        delta_weights_h_o += hidden_outputs[:, None] * output_error_term  # Weight step for hidden to output
        delta_weights_i_h += X[:, None] * hidden_error_term  # Weight step for input to hidden

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """Update weights with gradient descent step"""
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def train(self, features, targets):
        """Train the network"""
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            X = np.array(X, dtype=np.float64)    # CHANGE
            y = np.array(y, dtype=np.float64)    # CHANGE
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                       delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def run(self, features):
        """Run forward pass through the network"""
        features = np.array(features, dtype=np.float64)   # CHANGE
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs

# Hyperparameters
iterations = 5000
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1

