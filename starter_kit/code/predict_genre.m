function [ranks,info] = predict_genre(Xt_lyrics, Xq_lyrics, ...
                               Xt_audio, Xq_audio, ...
                               Yt)
% Returns the predicted rankings, given lyric and audio features.
%
% Usage:
%
%   RANKS = PREDICT_GENRE(XT_LYRICS, YT_LYRICS, XQ_LYRICS, ...
%                         XT_AUDIO, YT_AUDIO, XQ_AUDIO);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of lyric and audio features and produces a
% ranking matrix as explained in the project overview. 
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
N = size(Xq_lyrics, 1);
scores = zeros(N, 10);
Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));
Xq = bsxfun(@rdivide, Xq_lyrics, sum(Xq_lyrics, 2));

%D = kernel_intersection(Xt,Xq);
%[~, idx] = max(D, [], 2);
%ynn = idx(:, 1);
%yhat = Yt(ynn);
k = @(x,x2) kernel_poly(x,x2,3);
%Xt = full(Xt);
%Xq = full(Xq);
[yhat, info, prob] = kernel_libsvm(Xt,Yt,Xq,k)


[~, franks] = sort(prob, 2,'descend');
ranks = zeros(N,10);

for i = 1:N,
    for j = 1:10,
        ranks(i,j) = info.model.Label(franks(i,j));
    end
end
%for i=1:N
%    scores(i, yhat(i)) = 10;
    
%end

%ranks = get_ranks(scores);

end
