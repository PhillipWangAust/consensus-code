function [ L, T ] = binarizeProbDist( U, P )
% This function produces binary labels based on the maximization of
% agreement maximization. P is the prediction. U is the probability
% distribution across the labels
%   Detailed explanation goes here

nInst = size(U, 1);                 % Number of instances
nClasses = size(U, 2);              % Number of labels
nModels = size(P, 2) / nClasses;    % Number of models 
L = U;                              % Predicted values
T = zeros(nInst, 1);                % Threshold for each instances

for i = 1 : nInst
    % find the threshold for the case
    temp = U(i, :);
    [temp, ~] = sort(temp);
    threshold = (temp(1:nClasses-1) + temp(2:nClasses)) / 2;
    maxK = 0;
    maxTau = 0;
    for tau = [0 threshold 1];
        temp = U(i, :);
        temp(temp <= tau) = -1;
        temp(temp > tau) = 1;
        avgK = findAgreement(temp, P(i, :));
        if avgK > maxK
            maxTau = tau;
            maxK = avgK;
        end
    end
    L(i, U(i, :) <= maxTau) = -1;
    L(i, U(i, :) > maxTau) = 1;
    T(i) = maxTau;
end


end

