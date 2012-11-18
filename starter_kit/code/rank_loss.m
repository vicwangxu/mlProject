function [ rank_loss ] = rank_loss( ranks, YLabel )

%Caculte the rank loss of the result
loss = bsxfun(@eq,ranks,YLabel);
[~,rank_loss] = max(loss,[],2);
rank_loss = 1 - sum(1 ./ rank_loss) / numel(YLabel);


end

