function [ K ] = findKappaMat( X, Y)
%   Detailed explanation goes here

%   Find kappa value for X, Y Matrix
% X is the user value, Y is the consensus
A = sum(X == 1 & Y == 1);
B = sum(X == -1 & Y == -1);
C = sum(X == 1 & Y == -1);
D = sum(X == -1 & Y == 1);
T = A + B + C + D;
po = (A + B).*T;
pe = (A + C).*(A + D) + (B + C).* (B + D);
lId = po == pe;
K = (po - pe)./ (T.*T - pe);
K(lId) = 0;
end


