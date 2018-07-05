function [trainFeatures, trainLabels, testFeatures, testLabels] = loadDataset(DatasetName)
%% This function loads the dataset
%DatasetName = 'enron';

% Reading the dataset and label Count
trainFile = strcat('../../Data/', DatasetName, '/', DatasetName, '-train.arff');
testFile = strcat('../../Data/', DatasetName, '/', DatasetName, '-test.arff');
xmlFile = strcat('../../Data/', DatasetName, '/', DatasetName, '.xml');
xdoc = xmlread(xmlFile);
items = xdoc.getElementsByTagName('label');
classCount = items.getLength();
[~, ~, ~, trainData] = arffread(trainFile);
[~, ~, ~, testData] = arffread(testFile);
featureCount = size(trainData, 2) - classCount;

% obtaining features and labels
trainFeatures = trainData(:, 1 : featureCount);
trainLabels = trainData(:, featureCount + 1 : featureCount + classCount);
trainLabels(trainLabels~=0) = 1;

testFeatures = testData(:, 1 : featureCount);
testLabels = testData(:, featureCount + 1 : featureCount + classCount);
testLabels(testLabels~=0) = 1;

end
