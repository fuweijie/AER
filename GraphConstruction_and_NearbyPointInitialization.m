function [ZH,Z_arr,Real_Anchor_Idx_Arr,Arr_of_FinerAnchors,NearbySetMatrix]=GraphConstruction_and_NearbyPointInitialization(data,N_arr,s)
  curr_data=data;
  Z_arr=cell(1,length(N_arr)-1);
  Virt_Anchor_arr=cell(1,length(N_arr)-1);
  Real_Anchor_Idx_Arr=cell(1,length(N_arr));
 
  Neirhbor_Connect_Idx_arr=cell(1,length(N_arr)-1);
  Accum_Connect_arr=cell(1,length(N_arr)-1);
  
  Total_Real_Anchor_Idx=[];
  for no_level=(length(N_arr)):-1:2
      if N_arr(no_level)>5000 && N_arr(no_level)<=10000
         rand_id=randperm(N_arr(1));
         sub_data=data(rand_id(1:200000),:);
         [Virt_Anchor,~,~]=yael_kmeans(single(sub_data)',N_arr(no_level),'niter', 10,'nt',6);Virt_Anchor=double(Virt_Anchor)';
         [Real_Anchor_Idx]=find_Real_Anchor(data,Virt_Anchor);
      elseif N_arr(no_level)>10000
         rand_id=randperm(N_arr(1));
         Real_Anchor_Idx=rand_id(1:N_arr(no_level));     
         Virt_Anchor=data(Real_Anchor_Idx,:);
      else
        [Virt_Anchor,~,~]=yael_kmeans(single(data)',N_arr(no_level),'niter', 10,'nt',6);Virt_Anchor=double(Virt_Anchor)';
        [Real_Anchor_Idx]=find_Real_Anchor(data,Virt_Anchor);  
      end
        Real_Anchor_Idx=unique(Real_Anchor_Idx);
        Virt_Anchor_arr{1,no_level-1}= Virt_Anchor;  
    [Real_Anchor_Idx,~]=setdiff(Real_Anchor_Idx,Total_Real_Anchor_Idx);
    Total_Real_Anchor_Idx=[Total_Real_Anchor_Idx,Real_Anchor_Idx];
    Real_Anchor_Idx_Arr{1,no_level}=Real_Anchor_Idx;
  end
   Real_Anchor_Idx_Arr{1,1}=setdiff(1:N_arr(1,1),Total_Real_Anchor_Idx);
     
  for no_level=1:(length(N_arr)-1)
    Virt_Anchor=Virt_Anchor_arr{1,no_level};
    [temp_Z] = WeightEstimation_Batch_Approx(curr_data', Virt_Anchor', s);
    curr_data=Virt_Anchor;
    Z_arr{no_level}=temp_Z; 
  end
  
  for no_level=1:(length(N_arr)-1)
    if no_level==1
       ZH= Z_arr{no_level};
    else
       ZH= ZH * Z_arr{no_level}; 
    end
  end
  
  
  %NearbyPoint Index
  curr_data=data(Real_Anchor_Idx_Arr{1,1},:);
  for no_level=1:(length(N_arr)-1)
    Real_Anchor_Idx=Real_Anchor_Idx_Arr{1,no_level+1};
    Virt_Anchor=data(Real_Anchor_Idx,:);
    [temp_E] = WeightEstimation_Batch_Approx(curr_data', Virt_Anchor',1);
    if size(temp_E,2)<length(Real_Anchor_Idx)
        temp_E(end,length(Real_Anchor_Idx))=0;
    end
    curr_data=Virt_Anchor;
    [~,idx]=max(temp_E');
    Accum_Connect_arr{1,no_level}=temp_E;
    Neirhbor_Connect_Idx_arr{1,no_level}=Real_Anchor_Idx(idx);
  end
  
  %NearbyPoint IndexMatrix
  E1=Accum_Connect_arr{1,1}';E2=Accum_Connect_arr{1,2}';E3=Accum_Connect_arr{1,3}';
  self_E3=size(E3,1);
  self_E2=size(E3,2);
  self_E1=size(E2,2);
  self_E0=size(E1,2);
  E_temp=[speye(self_E3,self_E3),E3,E3*E2,E3*E2*E1;...
                sparse(self_E2,self_E3),speye(self_E2,self_E2),E2,E2*E1;...
                sparse(self_E1,self_E3+self_E2),speye(self_E1,self_E1),E1;...
                sparse(self_E0,self_E3+self_E2+self_E1),speye(self_E0,self_E0)];
  E_temp(E_temp>0.9999)=1;
  re_id=[Real_Anchor_Idx_Arr{1,4},Real_Anchor_Idx_Arr{1,3},Real_Anchor_Idx_Arr{1,2},Real_Anchor_Idx_Arr{1,1}];
  EE(re_id,:)=E_temp;  
  E_temp=EE;  
  EE(:,re_id)=E_temp;   
  NearbySetMatrix=EE;
  
  %Finer_Anchor IndexArr
  for no_level=(length(N_arr)-1):-1:1 
    Next_Finer_Element= Accum_Connect_arr{1,no_level}';
    current_coarse=Real_Anchor_Idx_Arr{1,no_level+1};
    Next_Finer_ID=Real_Anchor_Idx_Arr{1,no_level};
    for id_coarse=1: length(current_coarse)
        temp_element=Next_Finer_Element(id_coarse,:);
        FinerAnchors_Arr{1,current_coarse(id_coarse)}=Next_Finer_ID(temp_element>0);
    end
  end
  
  Arr_of_FinerAnchors=FinerAnchors_Arr;
end

function  [real_Idx]=find_Real_Anchor(data,anchor)
data=single(data)' ;
anchor=single(anchor)';
s=1;
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
val=val(:);
index=[];
for i=1:s
    index=[index,1:n];
end
Z=sparse([index,n],[ind;m],[val;0]);
[~,real_Idx]=max(Z);
end