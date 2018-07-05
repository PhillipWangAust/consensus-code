function [ model , featureMatrix] = trainLabels(P,L)
%   Returns a  linear model based on training of predictions based on machine labellers
%P denotes various machine predictions for all the instances for a single label .is a 2D matrix
%L denotes truth/human labeller consensus here is a column Vector /Response
%labelId is the current label
%   Detailed explanation goes here
%% get N number of models / predictions
instanceCount = size(L, 1); % instance count in training
%labelCount = size(L, 2);   % labels count 
%libLinearPath = '../../../liblinearinc-2.1/matlab';
%config.libLinearPath = libLinearPath;
%addpath(libLinearPath);

%disp('trainModelLinear Entered');


for i=1:instanceCount
	featureVector=P(:,i);
	featureVector=featureVector';
    if i == 1
        featureMatrix = featureVector;
    else
        featureMatrix = [featureMatrix;featureVector];  % append each feature vector to matrix
    end
		
end
lData = featureMatrix;
lLabels = L;
[oversampleData oversampleLabels]=oversample2(lData,lLabels);
%disp('oversampling done in trainModelLinear');
model=train(oversampleLabels,sparse(oversampleData),'-s 0 -C -v 3 -q');
c_opt=model(1);
%training with optimum c value for each label
model=train(oversampleLabels,sparse(oversampleData),sprintf('-s 0 -c %d -q',c_opt));
end


