
function [Z]=WeightEstimation_Batch_Approx(data,anchor,s)
data=single(data)  ;
anchor=single(anchor);
%% Deal with the labeled part via Zl
param_branching=32;
param_check=40;
[~,n]=size(data);
[~,m]=size(anchor);
index = flann_build_index(anchor,struct('algorithm', 'kmeans','branching',param_branching,'iterations',5,'cores',6));
[result,dists] = flann_search(index,data,s,struct('algorithm', 'kmeans','checks',param_check,'cores',6));
pos=result';val=dists';
clear dists result
ind=pos(:);
sigma = mean(val(:,s).^0.5);
val = exp(-val/(1/1*sigma^2));
val = repmat((sum(val,2)+eps).^-1,1,s).*val;  
val=val(:);
index=[];
for i=1:s
    index=[index,1:n];
end
Z=sparse([index,n],[ind;m],[val;0]);
