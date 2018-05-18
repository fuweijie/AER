 function     [F_sparse,Position]=Sparse_F_Top_k(F_past,N_class_kept)
 F_past(F_past<0)=0;   
[N_Can,C]=size(F_past);
 F_sparse = zeros(N_Can,C);
 val=zeros(N_Can,N_class_kept);
 pos = val;
 for iter_s = 1:N_class_kept
   [val(:,iter_s),pos(:,iter_s)] = max(F_past+eps,[],2);
   tep = (pos(:,iter_s)-1)*N_Can+[1:N_Can]';
   F_past(tep) = -1;
 end
 clear Dis;
 clear tep;
 ind = (pos-1)*N_Can+repmat([1:N_Can]',1,N_class_kept);
 Position=pos;
 F_sparse([ind]) = [val];  
 F_sparse=F_sparse./repmat(sum(F_sparse,2),1,C); 
 F_sparse=full(F_sparse);

