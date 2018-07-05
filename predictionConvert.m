function [ B ] = predictionConvert(A)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%   converts the matrix A which is original prediction to B
    
X = sum(A, 2);
X = (X == 0);
lId = (A == 0);
A(lId) = -1;
A(X, :) = 0;
B = A;

end

