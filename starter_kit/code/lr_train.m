function [w obj gradnorm] = lr_train(X, Y, C, varargin)
% Train a logistic regression model.
%
% Usage:
%
%   [W OBJ GRADNORM] = LR_TRAIN(X, Y, C, ...)
%   [W] = LR_TRAIN(X, Y, C, 'step_size', STEP, ...)
%   [W] = LR_TRAIN(X, Y, C, 'max_iter', T, ...)
%   [W] = LR_TRAIN(X, Y, C, 'stop_tol', TOL, ...)
%
% X is a N x P matrix of N examples with P features each.
% Y is a N x 1 vector of 0-1 class labels. C is the regularization
% parameter. Also accepts a number of optional arguments:
%    'step_size': The step size of gradient ascent. (Default: 0.1)
%    'stop_tol': The tolerance of the stopping condition. (Default: 0.001)
%    'max_iter': The maximum # of iterations to run. (Default: 1000)
%  
% Trains a logistic regression model with 1 additional
% bias parameter, which is prepended to the front of the 1 x (P+1) returned 
% weight vector W. Uses gradient ascent with the specified parameters. 
% The second return value OBJ is a vector where OBJ(t) is the LR objective
% (conditional log likelihood) at round t. The third return value GRADNORM
% is the average norm of the gradient at the final iteration.
%
% SEE ALSO
%   LR_TEST, LR_GRADIENT

% Parse VARARGIN for optional arguments, consisting of 'PROPERTY', VALUE pairs.
% NOTE: There are many ways of doing this, here is one the TA wrote many years ago. 
defaults.step_size = 0.03;
defaults.stop_tol = 1e-4;
defaults.max_iter = 600;

opts = propval(varargin, defaults);

% Add a constant feature to each example to learn a bias term.
X = [ones(size(X,1), 1) X];

% Initialize all weights to 0
w = zeros(1, size(X,2));

% *** NOTE: Change Y from 0,1 to -1, 1 to simplify training expression, and
% match the lecture notes online.
Y(Y==0) = -1;

% *** NOTE: In your code, to compute the LR objective with larger step
% sizes you need to take steps to avoid numerical overflow. E.g., if
% exp(-y * X * w) is Infinity, then you need to clamp it to the 'realmax'
% value to avoid getting log(Inf) = Inf.

% YOUR CODE GOES HERE
t = CTimeleft(opts.max_iter);
for i = 1: opts.max_iter

t.timeleft();
    %caculate the x_times_y_times_w
x_times_y = bsxfun(@times,X,Y);
e = exp(- x_times_y * w');
e(e == Inf) = realmax;

condition_log = -log(1 + e);
obj(i) = sum(condition_log);



grad = lr_gradient(X,Y,w,C);
w = w + opts.step_size * grad;
gradnorm = sqrt(sum(grad .^ 2)) / size(X,1);
if(i > 2)
    if gradnorm < opts.stop_tol
    break;
    end
end


end
