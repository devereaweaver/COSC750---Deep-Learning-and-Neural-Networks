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
iterations = 30000
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1

