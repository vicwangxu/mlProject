function K = kernel_intersection(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

%normalize the frequency
X = bsxfun(@rdivide, X, sum(X,2));
X2 = bsxfun(@rdivide, X2, sum(X2,2));

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

%transpose X, X2 to increase the speed
X = X';
X2 = X2';

%caculate the kernel
t = CTimeleft(m);
for i = 1:m
    %caculate the min
    t.timeleft();
    result = bsxfun(@min,X2(:,i),X);
    result = sum(result);
    
    K(i,:) = result;
end

