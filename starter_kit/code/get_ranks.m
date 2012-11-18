function R = get_ranks(yscores)
% Returns a ranking of class labels based on the given scores.
%
% Usage:
%
%   R = GET_RANKS(YSCORES)
%
% Sorts the scores in descending order to obtain the correct ranking. NOTE:
% this function depends on MATLAB's sort function, which does not break
% ties randomly. If two scores are equal, the smaller index will always be
% given the better rank. If you wish to break ties in a different fashion,
% you will need to implement this.

%give the score according to their popularity in the training set
pop = [0.2 0.7 0.3 0.8 0.9 0.4 0.5 0.6 0.1 0];
yscores = bsxfun(@plus,yscores,pop);


[~, R] = sort(yscores, 2, 'descend');

end