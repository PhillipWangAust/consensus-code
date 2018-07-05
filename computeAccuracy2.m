function [ f ] = computeAccuracy(P,L)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% L is the original set of labels
% P is the prediction

index = sum(P, 2) ~= 0;
index = index .* [1:size(P,1)]';
index = index(index~=0);
P = P(index, :);
L = L(index, :);
% P is the prediction

TP = sum(sum((L==1) .* (P==1)));
FP = sum(sum((L==-1) .* (P==1)));
FN = sum(sum((L==1) .* (P==-1)));
TN = sum(sum((L==-1) .* (P==-1)));
fprintf('TP is %d\n',TP);
fprintf('FP is %d\n',FP);
fprintf('FN is %d\n',FN);
fprintf('TN is %d\n',TN);
f=(TP+TN)/(TP+FP+TN+FN);
end
