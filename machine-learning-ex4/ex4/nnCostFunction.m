function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Cost main
% Loop through all examples row
% Forward propagate
% Get the predicted value
% Calculate cost for that predicted value

% Take matrix of values
a1 = [ones(size(X,1),1) X];

% Calculate linear combination of neurons value for each sample, where column
% value is the value for that neuron, row is sample
%
% So row 2, col 3 is z value for z value for 2nd sample's 3rd neuron on 2nd layer
z2 = a1 * (Theta1');

% Calculate the sigmoid function component wise
% Gives us matrices of a2 for each sample
a2 = [ones(size(z2,1),1) sigmoid(z2)];

% Repeat
z3 = a2 * (Theta2');

% h_theta aka
% a3 is matrix where row i is predicted value for sample i
a3 = sigmoid(z3);

% y is m x K matrix
% Loop through row, col of y
% sum z3(row)

% Unpack y vector into matrix of node output layer outputs
yMatrix = zeros(m, num_labels);
for i = 1:m
    yValue = y(i);
    yMatrix(i, yValue) = 1;
end

% Jmatrix = [ y .* log(a3) ] + [ (1 - y) * log(1 - a3)]
% Row is for sample i, col is for output neuron col, the cost of that
% Need to sum everything all together
Jmatrix = (-yMatrix .* log(a3)) - ((1-yMatrix) .* log(1 - a3));

% Cost regularized
% Component wise square since "rolled out" into vector
% Sum
Theta1Squared = (Theta1 .^ 2);
Theta1Regularized = sum(sum(Theta1Squared(:,2:end)));

Theta2Squared = (Theta2 .^ 2);
Theta2Regularized = sum(sum(Theta2Squared(:,2:end)));

Jregularized = (lambda / (2 * m)) * (Theta1Regularized + Theta2Regularized);

J = ((1/m) * sum(sum(Jmatrix))) + Jregularized;


% Calculate gradients
% m x num_labels

% Error of output at each input x_i, where i is row number.
% This is a matrix of the gradient at all points
d3 = a3 - yMatrix;

% Sums the errors for each node over all samples where
% for (i, j), row i is for sample x_i, and how much back error it
% contributes to neuron j
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

% Want to sum up all the little errors pushed back on to neuron j in
% layer {1,2}, multiple by al {1,2} because it is coeff of partial
% derivative for that line back
%
% j is neuron j in layer l, i is total push back from neuron i in layer l + 1
% this gives us a total error weight over all data points going from
% [neuron j, layer l] to [neuron i, layer l + 1]
D2 = d3' * a2;
D1 = d2' * a1;

Theta2_grad = (1/m) * D2;
% Regularize, so ignore first column as that's just bias so no derivative
% since it's constant. The coeff is regulariation param divided by average.
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2(:,2:end));

Theta1_grad = (1/m) * D1;
% Regularize, so ignore first column as that's just bias so no derivative
% since it's constant. The coeff is regulariation param divided by average.
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1(:,2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
