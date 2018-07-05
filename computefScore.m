function [ f ] = computefScore(P,L)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% L is the original set of labels
% P is the prediction

index = sum(P, 2) ~= 0;
index = index .* [1:size(P,1)]';
index = index(index~=0);
P = P(index, :);
L = L(index, :);
L(L==0) = -1;
P(P == 0) = -1;
%disp('hello in computefScore');
%disp(size(P));
%disp(size(L));

f=findFScore2(P,L,1);
end
