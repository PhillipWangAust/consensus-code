function [ threshold ] = findOptimumThreshold( U, P )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

nInst = size(U, 1);
nClasses = size(U, 2);
nModels = size(P, 2) / nClasses;

threshold = zeros(nInst, 1);
for i = 1 : nInst
    temp = U(i, :);
    [temp, ~] = sort(temp);
    thresholdTemp = (temp(1:nClasses-1) + temp(2:nClasses)) / 2;
    thresholdTemp = [0 thresholdTemp 1];
    maxK = 0;
    maxTau = 0;
    for tau = thresholdTemp
        temp = U(i, :);
        temp(temp <= tau) = -1;
        temp(temp > tau) = 1;
        sumK = 0;
        count = 0;
        for j = 1 : nModels
            [KVal, ind] = findKappaVec(P(i, (j - 1) * nClasses + 1 : j * nClasses), temp);
            if ind == 1
                count = count + 1;
                sumK = sumK + KVal;
            end
        end
        sumK = sumK/count;
        if sumK > maxK
            maxTau = tau;
            maxK = sumK;
        end
    end
    threshold(i) = maxTau;
end


end

