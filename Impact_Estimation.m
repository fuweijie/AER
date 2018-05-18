function [Impact_Matrix,Alpha,Beta]=Impact_Estimation2(Shared_Var_arr,M_past,Z,Candidate_idx,C)
Z_s_all=Z(Candidate_idx,:);
rW=Shared_Var_arr{1,1};
YlZl=Shared_Var_arr{1,2};
A=M_past*YlZl';
Alpha=M_past*Z_s_all'; % nh*nq
Alpha=sum(Z_s_all'.*Alpha);  % 1*nq
Beta=(Alpha+1).^(-1); % 1*nq

Gamma_Inter=M_past*rW*M_past; %nh*nh
Gamma=Gamma_Inter*Z_s_all'; %nh*nq
Gamma=sum(Z_s_all'.*Gamma); %1*nq

VarPhi=Z_s_all*A; %nq*c

Phi=Z_s_all*(M_past*(rW*A)); %nq*c

CrossPhi=sum(Phi'.*VarPhi'); %1*Nq

Delta=trace(A'*rW*A); %1*1

Term1=repmat((1-2*Alpha.*Beta+(Alpha.^2).*(Beta.^2)).*Gamma,C,1)...
    +repmat(2-2*Alpha.*Beta,C,1).*(Phi')...
    +repmat(2*Alpha.*(Beta.^2).*Gamma-2*Beta.*Gamma,C,1).*(VarPhi')...
    +Delta-2*repmat(Beta.*CrossPhi,C,1)+repmat((Beta.^2).*Gamma.*sum(VarPhi'.*VarPhi'),C,1);
Term2=Delta+Phi'-repmat(Alpha.*Beta,C,1).*Phi'-repmat(Beta.*CrossPhi,C,1);
Term3=Delta;
Impact_Matrix=Term1-2*Term2+Term3;
Impact_Matrix=Impact_Matrix';