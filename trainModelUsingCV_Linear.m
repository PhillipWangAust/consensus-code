function [ modelMatrix ] = trainModelUsingCV( trainData, trainLabel, CVFold )
%   Returns a matrix of SVM models for each of the labels
%   Detailed explanation goes here
%add liblinear path to use library
libLinearPath = '../../liblinear-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);

%% Parameters
k = CVFold;    
beta = 1;

%% get N number of models / predictions
instanceCount = size(trainData, 1); % instance count in training
labelCount = size(trainLabel, 2);   % labels count 

data = trainData;
label = trainLabel;

%% select CV folds
CV = cvpartition(instanceCount, 'KFold', k);

%% Train SVM on each partition and each label
%CArr = zeros(labelCount, 1);
%GammaArr = zeros(labelCount, 1);
featureCount = size(trainData, 2);

for l = 1 : labelCount
    %CArr(l) = -1; GammaArr(l) = -5; fMax = 0;
    %lowC = 4; highC = 8;
    %lowG = 2 - log2(featureCount); highG = 3 - log2(featureCount);
    %for c = lowC : highC
        %for g = lowG : highG
            %config.C = c;
            %config.gamma = g;
            %fAvg = 0;
                %trainDataCV = data(CV.training(i), :);
                %trainLabelCV = label(CV.training(i), l);
                %testDataCV = data(CV.test(i), :);
                %testLabelCV = label(CV.test(i), l);

                % add non zero row to training data
                %nzRow = find(trainLabelCV, 1);

                % train SVM
                if size(nzRow, 1) * size(nzRow, 2) == 0
                    nzRow = 1;
                end
                model = train([trainDataCV(nzRow, :); trainDataCV], [trainLabelCV(nzRow); trainLabelCV], config);

                % prediction for training data
                %[predictionLabelTrn, accuracyTrn, ~] = svmpredict(trainLabelCV, trainDataCV, model);
                [predictionLabelCV, ~, ~] = svmpredict(testLabelCV, testDataCV, model);
                f = findFScore(predictionLabelCV, testLabelCV, beta);
                fAvg = fAvg + f;
            end
            
            fAvg = fAvg / k;
            if fAvg > fMax
                fMax = fAvg;
                CArr(l) = config.C;
                GammaArr(l) = config.gamma;
            end
        end
    end

    % add non zero row to training data
    nzRow = find(label(:, l), 1);
    
    if size(nzRow, 1) * size(nzRow, 2) == 0
        nzRow = 1;
    end
    % train SVM on the optimum parameters
    config.C = CArr(l);
    config.gamma = GammaArr(l);
    model = trainSVM([data(nzRow, :); data], [label(nzRow, l); label(:, l)], config);   % train the model on the entire data
    if l == 1
        modelMatrix = model;
    else
        modelMatrix = [modelMatrix model];  % append each label model to the matrix
    end
end

end

