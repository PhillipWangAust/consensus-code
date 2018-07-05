function [ P ] = predictLabels( modelMatrix, testData )
% This function predicts the labels for each instance using the models for
% each of the labels
%   Detailed explanation goes here

labelCount = size(modelMatrix, 2);
instanceCount = size(testData, 1);
P = zeros(instanceCount, labelCount);
parfor l = 1 : labelCount
    [P(:, l), ~, ~] = svmpredict(P(:, l), testData, modelMatrix(l)); 
end


end

