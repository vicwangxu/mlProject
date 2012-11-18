function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the Guassian kernel
% with parameter sigma=20.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

%transpose x to increase the speed
X = X';
X2 = X2';

for i = 1:m
    % caculate the distance between the two example
    result = bsxfun(@minus,X2(:,i),X);
    dis = result .^ 2;
    dis = sum(dis,1); 
    K(i,:) = exp(- dis ./ (2*sigma^2) );    
end

