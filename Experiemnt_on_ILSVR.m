
load('Data_ILSVR2012_FC7') 
N0=length(label);   
Start_Num=2;
T=500;
beta=0.01; s=3;
Param.TopK_Var=40;
N1=100000;
N2=10000;
N3=2500;
N_arr=[N0,N1,N2,N3];
flag_virtual=0;

Rand_Label=randperm(length(label));
Rand_Label=Rand_Label(1:Start_Num);

[ZH,Z_arr,Arr_of_Anchor_Idx,Arr_of_FinerAnchors,NearbySetMatrix]=GraphConstruction_and_NearbyPointInitialization(data,N_arr,s);
ZH_data=ZH(1:N0,:);  Z01_data=Z_arr{1,1};
temp=ZH_data'*Z01_data;
Delta=temp*Z_arr{1,2}*Z_arr{1,3};
Lambda=sparse(1:size(Z01_data,2),1:size(Z01_data,2),sum(Z01_data).^(-1));
rL=Delta-temp*Lambda*temp';
rL=rL+eye(size(rL))*1e-6;
clear temp
  
%% Random Sampling
rand_idx=randperm(N0);
init_label_index=rand_idx(1:Start_Num);

ran_num=(Start_Num-1):1:T;tic
class_repeat=3;
Error_RS=zeros(class_repeat,length(T));
for rep=1:class_repeat
  idx_rand=[init_label_index,randperm(N0)];
  for i=1:length(ran_num)  
     random_real_select_ind=idx_rand(1:ran_num(i));
     [err,~,~,~,~] =Regularization(ZH_data, rL, label', random_real_select_ind, beta);
     Error_RS(rep,i)=err;
  end
end

%% MAER 
Param.NumQuery=T;
Param.rL=rL;
Param.label=label;
Param.label_index=init_label_index;
Param.beta=beta;
Param.Shared_Var_arr=Delta;
Param.NearbySetMatrix=NearbySetMatrix;
Param.Arr_of_FinerAnchors=Arr_of_Nearby_Finer;

Param.Real_IDX_Set=Arr_of_Anchor_Idx{1,end};
Error_MAER=AL_MAER(ZH_data,Param);







 
