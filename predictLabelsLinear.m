function  P = predictLabelsLinear( modelMatrix, testData )
% This function predicts the labels for each instance using the models for
% each of the labels
%   Detailed explanation goes here
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
labelCount = size(modelMatrix, 2);
instanceCount = size(testData, 1);
P = zeros(instanceCount, labelCount);
for l = 1 : labelCount
    [P(:, l), ~, ~] = predict(P(:, l), sparse(testData), modelMatrix(l),'-b 1');
    %disp('in predictLabelsLinear');
    %disp(size(P(:,l))); 
end


end

