function error=AL_MAER(ZH_data,Param)
NumQuery=Param.NumQuery;
TopK_Var=Param.TopK_Var;
rL=Param.rL;
NearbySetMatrix=Param.NearbySetMatrix;
label=Param.label;
label_index=Param.label_index;
beta=Param.beta;
Real_IDX_Set=Param.Real_IDX_Set;
Shared_Var_arr{1,1}=Param.Shared_Var_arr;
Next_FinerAnchors=Param.Arr_of_FinerAnchors;
C=max(label);
N0=length(label);
Start_Num=length(label_index) +1;

error=zeros(1,NumQuery);
[err,F,Nor_F,M_past,ZlYl] =Regularization(ZH_data, rL, label', label_index, beta);
Shared_Var_arr{1,2}=ZlYl;
error(Start_Num-1)=err;   

  for no_select=Start_Num:NumQuery
    Nor_F_can=Nor_F(Real_IDX_Set,:);
    [Nor_F_Sparse,Position_can]=Sparse_F_Top_k(Nor_F_can,2);
    Virtual_IDX_Set=1:length(Real_IDX_Set);
    [Raw_Impact,Virtual_var_Alpha,Virtual_var_Beta]=Impact_Estimation(Shared_Var_arr,M_past,ZH_data,Real_IDX_Set,C);       
    Raw_Impact=sum(Raw_Impact.*Nor_F_Sparse,2);
    E=NearbySetMatrix(Real_IDX_Set,:);
    [~,order] = max(Nor_F,[],2);   output = order';
    Y = zeros(N0,C);
    for tp_c = 1:C
         ind = find(output == tp_c);
         Y(ind',tp_c) = 1;
    end 
    Error_Points=sum(abs(Y-Nor_F).^2,2);    
    Error_Nearbyset=E*Error_Points;
    Ave=mean(Error_Points); 
    
    Total_Estimated_Value=(Error_Nearbyset).^(1-Ave).*Raw_Impact.^(Ave);
    [~,IDX_sort]=sort(Total_Estimated_Value,'descend'); 
    Real_Top_Node=Real_IDX_Set(IDX_sort(1:1));
    label_index=[label_index,Real_Top_Node]; 
    [err,F,Nor_F,M_past,ZlYl] =Regularization(ZH_data, rL, label', label_index, beta);
    error(no_select)=err;     
    Shared_Var_arr{1,2}=ZlYl;
    Real_IDX_Set(IDX_sort(1:1)) =[];  
    Real_IDX_Set=[Real_IDX_Set,Next_FinerAnchors{1,Real_Top_Node}]; 
  end 
  
end
