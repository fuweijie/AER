function [err,F,Nor_F,M_past,ZlYl] = Regularization(Z, rL, ground, label_index, beta)
C=max(ground);
[n,m] = size(Z);
ln = length(label_index);
Yl = zeros(ln,C);
for i = 1:C
    ind = find(ground(label_index) == i);
    Yl(ind',i) = 1;
    clear ind;
end

Zl = Z(label_index',:);
LM =beta*rL+ Zl'*Zl;
for i=1:m
    LM(i,i)=LM(i,i)+1e-6;
end
RM = Zl'*Yl;  ZlYl=RM';
M_past=inv(LM);
A = M_past*RM; 
F=Z*A;

F1 = F*diag((sum(F)+eps).^-1); 
[~,order] = max(F1,[],2);
output = order';
clear order;
output(label_index) = ground(label_index);
err = length(find(output ~= ground))/(n);

Nor_F=F;
Nor_F(Nor_F<0)=0;
Nor_F = Nor_F*diag((sum(Nor_F)+eps).^-1);
Nor_F=  repmat(sum(Nor_F')'.^-1,1,C).*Nor_F;