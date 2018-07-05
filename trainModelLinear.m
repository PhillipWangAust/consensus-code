function [ modelMatrix ] = trainModelLinear( trainData, trainLabel )
%   Returns a matrix of linear models for each of the labels
%   Detailed explanation goes here
%% get N number of models / predictions
instanceCount = size(trainData, 1); % instance count in training
labelCount = size(trainLabel, 2);   % labels count 
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
data = trainData;
label = trainLabel;
%disp('trainModelLinear Entered');

for l = 1 : labelCount
 %fprintf('got to label %d\n',l);
    lData = data;
    lLabels = label(:, l);
    [oversampleData oversampleLabels]=oversample2(lData,lLabels);
    %disp('oversampling done in trainModelLinear');
    model=train(oversampleLabels,sparse(oversampleData),'-s 0 -C -v 3 -q');
    c_opt=model(1);
    %training with optimum c value for each label
    model=train(oversampleLabels,sparse(oversampleData),sprintf('-s 0 -c %d -q',c_opt));
    if l == 1
        modelMatrix = model;
    else
        modelMatrix = [modelMatrix model];  % append each label model to the matrix
    end
 %fprintf('finished label %d\n',l);
end
%disp('trainModelLinear exited');
end


