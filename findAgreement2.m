function [ K ] = findAgreement2( L, P)
% find the agreement between users on a instance using average kappa 
%   Detailed explanation goes here

nClasses = size(L, 2);
nModels = size(P, 2) / nClasses;


LRep = repmat(L, 1, nModels);
K = findKappaVec(P, LRep);

end

