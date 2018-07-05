function [ f ] = findFScore( P, L, beta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% L is the original set of labels
% P is the prediction

TP = sum(sum((L==1) .* (P==1)));
FP = sum(sum((L==0) .* (P==1)));
FN = sum(sum((L==1) .* (P==0)));
precision = TP/(TP + FP);
recall = TP / (TP + FN);
f = (1 + beta^2)*(precision * recall)/(beta^2 * precision + recall);
if TP == 0
    f = 0;
end

end

