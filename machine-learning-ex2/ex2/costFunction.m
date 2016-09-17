function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

x = X';
thetaTx = theta' * x;			% 0'x
hThetax = sigmoid(thetaTx);		% Sigmod of thetaTx
errors1 = y'*log(hThetax)';
errors2 = (1 .- y)' * log(1 .- hThetax)';
J = -1 * (sum(errors1 + errors2))/m;

for feature_n = 1: length(theta)
	grad(feature_n) = sum((hThetax - y')*X(:,feature_n)) / m;
end

% =============================================================

end
