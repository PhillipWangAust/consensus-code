function [ K ] = findAgreement( L, P)
% find the agreement between users on a instance using average kappa 
%   Detailed explanation goes here

nClasses = size(L, 2);
nModels = size(P, 2) / nClasses;
K = 0;
count = 0;

for i = 1 : nModels
    [KVal, ind] = findKappaVec(P((i - 1) * nClasses + 1 : i * nClasses), L);
    if ind == 1
        count = count + 1;
        K = K + KVal;
    end
end

K = K / count;

end

