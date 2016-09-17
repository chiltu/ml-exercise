function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% Cost Calculations

%%%%%%%%%%%%%%%%%%%%%%%%%%% WORKING WITH POR LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x = X';
% thetaTx = theta' * x;			% 0'x
% hThetax = sigmoid(thetaTx);		% Sigmod of thetaTx
% errors1 = y'*log(hThetax)';
% errors2 = (1 .- y)' * log(1 .- hThetax)';

% % Regularization calculations
% thota = theta(2:length(theta),:);
% sumthetasquared = sum(thota .* thota);

% % Cost J(0)
% J = (-1 * (sum(errors1 + errors2))/m) + (lambda*sumthetasquared)/(2*m);


% grad(1) = sum((hThetax - y')*X(:,1)) / m;
% for feature_n = 2: length(theta)
% 	grad(feature_n) = (sum((hThetax - y')*X(:,feature_n)) / m) + lambda*theta(feature_n)/m;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%% END WORKING WITH POR LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cost Calculations
x = X';
thetaTx = theta' * x;			% 0'x
hThetax = sigmoid(thetaTx);		% Sigmod of thetaTx
errors1 = y'*log(hThetax)';
errors2 = (1 .- y)' * log(1 .- hThetax)';

% Cost J(0)
J = (-1 * (sum(errors1 + errors2))/m) + (lambda*(sum(theta(2:end).^2)))/(2*m);

% Calucation of gradient using the vecto implementation
grad = (X' * (hThetax' - y))/m;
temp = theta;
temp(1) = 0;
grad = grad + ((lambda/m) .* temp);

% =============================================================

end
