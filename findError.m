function [ err ] = findError(A, B)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

err = 100*sum(sum(A~=B))/(size(A,1)*size(A,2));

end

