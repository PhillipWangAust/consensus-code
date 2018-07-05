function [ k ] = findKappaColumnVec(X, Y)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%   Find kappa value for X, Y vectors
% X is the user value, Y is the consensus
a = sum(X == 1 & Y == 1);
b = sum(X == -1 & Y == -1);
c = sum(X == 1 & Y == -1);
d = sum(X == -1 & Y == 1);
t = a + b + c + d;
po = (a + b) * t;
pe = (a + c) * (a + d) + (b + c) * (b + d) ;
k = (po - pe) / (t*t - pe);
end

