function [ k, temp ] = findKappaVec(X, Y)
%   Find the kappa Value between 2 vectors
%   Detailed explanation goes here

%   Find kappa value for X, Y vectors
% X is the user value, Y is the consensus
a = sum(X == 1 & Y == 1);
b = sum(X == -1 & Y == -1);
c = sum(X == 1 & Y == -1);
d = sum(X == -1 & Y == 1);
t = a + b + c + d;
po = (a + b) * t;
pe = (a + c) * (a + d) + (b + c) * (b + d);

if po == pe
    k = 0;
else
    k = (po - pe) / (t*t - pe);
end

temp = 1;
if (sum(X == 1) + sum(X == -1) == 0) || (sum(Y == 1) + sum(Y == -1) == 0)
    temp = 0;
end

k = (k+1)/2;

end

