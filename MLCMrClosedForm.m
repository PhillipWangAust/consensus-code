function [ U, Q ] = MLCMrClosedForm( nInstance, nClass, nModel, A, alpha, B )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% This function uses the MLCM-r method to establish consensus using the
% data in form of instance vs group prediction which would be sent as
% parameter :-
% A     -> instance x group matrix
% alpha -> relaxation parameter
% B     -> the original label of the group nodes
%Dataset='bibtex';
%fId = fopen('MLCMClosedForm.txt','a');
%fprintf(fId,'\n--------------------------------------\n');
%fprintf(fId, ' %s \n', Dataset);
U = 1 / nClass * ones(nInstance, nClass);
Q = zeros(nClass * nModel, nClass);

Dn = diag(1./sum(A, 2));
Dv = diag(1./(sum(A, 1)+0.001));
Dl = diag(1./(sum(A, 1) + alpha))*diag(sum(A,1));
Doml = diag(alpha./(sum(A, 1) + alpha));
%tic;
S= Dv * A' * Dn * A;
%fprintf(fId,'Time in seconds for calculation of S is %f\n',toc);
%tic;
temp =inv(eye(nModel * nClass) - Dl * S);
%fprintf(fId,'Time in seconds for calculation of inverse is %f\n',toc);
%tic;
Q = temp * Doml * B;
%fprintf(fId,'Time in seconds for calculation of Q is %f\n',toc);
%tic;
U = Dn * A * Q;
%fprintf(fId,'Time in seconds for calculation of U is %f\n',toc);
end
