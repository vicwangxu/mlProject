function [grad] = lr_gradient(X, Y, w, C)
% Compute the Logistic Regression gradient.
%
% Usage:
%
%    [GRAD] = LR_GRADIENT(X, Y, W, C)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of (-1, +1) class labels. W is a 1 x P weight vector. C is the regularization
% parameter. Computes the gradient w.r.t. W of the regularized logistic
% regression objective and returns a 1 x P vector GRAD.
%
% SEE ALSO
%   LR_TRAIN, LR_TEST

% YOUR CODE GOES HERE

y_times_x = bsxfun(@times,X,Y);
p_y_given_x = 1 ./ (1 + exp(-y_times_x * w'));
grad_mat = bsxfun(@times, y_times_x, (1 - p_y_given_x));
%grad_mat = bsxfun(@minus, grad_mat, C * w);
grad = sum(grad_mat,1) - C * w;