function [ partition ] = make_part(X,group)

%make partition for cross valiadation
totalSize = size(X,1);
groupSize = group;

partition =  ceil(groupSize * randperm(totalSize)/totalSize);

end

