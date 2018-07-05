%% This script is for generating different models using CV on train data
% The models are trained on different dataset. 

%% setup configuration
N_human = 6;                     % N_human human models are created
N_machine=5;			% N_machine models to be created
DatasetName = 'slashdot';
k = 5;                     % k fold CV is done for training the models
experimentTotal = 5;
libLinearPath = '../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
Folder = '../Output/modelsMachines/';
Folder1='../Output/modelsBias/';
fId = fopen(strcat(Folder, 'outputs_', DatasetName, '.txt'), 'a');
fprintf(fId, '\n-----------------------------------------------------\n');
alpha=1;
%% loading the data from the dataset
[trainData, trainLabel, testData, testLabel] = loadDataset(DatasetName);
disp(size(trainData));
trainData1 = trainData(1:200, :);
disp(size(trainData1));
% testData = testData(100 : 200, :);
% trainLabel = trainLabel(100 : 200, :);
% testLabel = testLabel(100 : 200, :);
flipprob=0.25;
%get trainLabel L from consensus of 
for expNum=1:1
	models= ones(N_human, 1);
%models(1:5) = 0;
	modelIndex = [1:N_human]';
%create subset of Labels for each model
	P=load([Folder1, DatasetName,'_model_1.y.1']);
	P=P(1:200,:);
	[nInst, nClasses] = size(P);
	fprintf('size of p is %d %d\n',nInst,nClasses);
	nModels = sum(models);
	P=zeros(nInst, nClasses*nModels);       % connection matrix
	i = 0;
        for j=1:N_human
            if models(j) == 1
                i = i + 1;
                temp = load([Folder1, DatasetName, '_model_', int2str(j), '.y.', int2str(expNum)]);
		temp=temp(1:200,:);
		[temp_1 temp_2]=size(temp);
		fprintf('size of temp is %d %d\n',temp_1,temp_2);
                P(:,(i-1)*nClasses+1:i*nClasses) = predictionConvert(temp);
            end
        end

    % preprocess the data to remove unpredicted instances 
        d = sum(P~=0, 2);
        Inst = [1:nInst]';

        P = P(d~=0, :);         % filter unpredicted rows
        Inst = Inst(d~=0, :);

        oldInst = nInst;
        nInst = size(P, 1);
	fprintf('ninstaces is %d\n',nInst);

        A = zeros(nInst, nClasses * nModels);   % Connection matrix with -1 replaced by 0
        A = P;
        %P(P==0) = -1;
        lId = A(:,:) == -1;
        A(lId) = 0;

        %% Run MLCM model
        % Using MLCM closed form
        % label matrix for group nodes
        L = eye(nClasses);
        B = repmat(L, nModels, 1);

        % obtain the consensus probability distribution
        %[U, Q] = MLCMr(nInst, nClasses, nModels, A, alpha, B);

        % use OnlineCM class to get U and Q
	oc_human=OnlineCM(A,B,alpha);
        fprintf('MLCM prediction done\n');
	L= oc_human.binarizeProbDist2(oc_human.U,oc_human.A);

        fprintf('Binarization Done\n');
	fprintf('size of L is %d\n',size(L,1));
end
trainLabel1=L;
labelCount = size(trainLabel1, 2);
instanceCount = size(trainLabel1, 1);

annotatorLabels=zeros(N_machine+1,instanceCount,labelCount);
modelSet = cell(labelCount, 1);
flipProb=0.25;
for expNum = 1 : 5
        fprintf(fId,'experimentnum=%d\n',expNum);
        fprintf(fId, '\n-----------------------------------------------------\n');

        % create a model for each label
        for l = 1 : labelCount
                lData = trainData1;
                lLabels = trainLabel1(:, l);
                %train a model for each label using liblinear
                M = train(lLabels, sparse(lData), '-s 0 -c 1');
                modelSet{l} = M;
                fprintf(fId, '%d Model trained\n', l);
                %predict probability distributions on whole data and write them to intermediate model files
                [predict_label, accuracy, prob_estimates] = predict(lLabels,sparse(lData),M, '-b 1');
                %dlmwrite([Folder, DatasetName, '_labelmodel_', int2str(l), '.y.', int2str(expNum)],prob_estimates, 'delimiter', '\t');
                fprintf(fId, 'Prediction Done for dataset  %s and label %d \n', DatasetName,l);
                cluster=kmeans(prob_estimates(:,1),N_machine);
                fprintf(fId, 'Clusters created for dataset %s and label %d \n', DatasetName,l);
                for i = 1 : instanceCount
                    for j= 1: N_machine+1
                        annotatorLabels(j,i,l)=getLabelOfAnnotator(j,cluster(i),flipProb,lLabels(i));
                    end
                end
                fprintf(fId, 'Annotator labels  created for dataset %s and label %d \n', DatasetName,l);



        end
        for j=1:N_machine+1
                dlmwrite([Folder, DatasetName, '_model_', int2str(j), '.y.', int2str(expNum)],squeeze(annotatorLabels(j,:,:)), 'delimiter', '\t');
                fprintf(fId,'written to file for anootator %d\n',j);
        end
end
