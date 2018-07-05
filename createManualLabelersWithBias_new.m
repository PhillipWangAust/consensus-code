%% This script is for generating different manual labelers with different ability on different labels
% The models are trained on different dataset. 
% First 1 to 5 models are kept are manual labelers in each dataset.


%% setup configuration

function [train_data train_label test_data test_label]= createManualLabelerswithBias_new(Dataset,N_human,N_exp,fraction,flipProbability)
N = N_human;                     % N models/annotators are created
DatasetName = Dataset;
expNum=N_exp;
%add liblinear path to use library
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
%create output folders
Folder = '../../Output_new/modelsHuman/';
fId = fopen(strcat(Folder, 'outputs_', DatasetName, '.txt'), 'a');
fprintf(fId, '\n-----------------------------------------------------\n');
fprintf(fId,'%s\n',DatasetName);

%% loading the data from the dataset
[trainData, trainLabel, testData, testLabel] = loadDataset(DatasetName);


%% Partition data for training and testing
data = [trainData;testData];
labels = [trainLabel;testLabel];
total_label_count = size(labels, 2);
total_instance_count = size(labels, 1);

%create a partition of training and testing data
frac=fraction;
train_instance_count=frac*total_instance_count;
train_instance_count=uint64(train_instance_count);
train_label_count=total_label_count;
fprintf('total_instances are %d\n',total_instance_count);
fprintf('training_instances are %d\n',train_instance_count);
annotatorLabels=zeros(N,train_instance_count,train_label_count);
modelSet = cell(train_label_count, 1);
flipProb=flipProbability;
fprintf(fId,'createManual Labelers experimentnum=%d\n',expNum);
fprintf(fId, '\n-----------------------------------------------------\n');
train_index=randsample(total_instance_count,train_instance_count);
train_data=data(train_index,:);
train_label=labels(train_index,:);
temp=zeros(total_instance_count,1);
temp(train_index)=1;
test_data=data(temp==0,:);
test_label=labels(temp==0,:);
cluster_count=5;
	% create a model for each label
for l = 1 : train_label_count
	lData = train_data;
	lLabels = train_label(:, l);
	%train a model for each label using liblinear
	 %to train using crossfold validation..one time job
	M=train(lLabels,sparse(lData),'-s 0 -C -v 5');
	c_opt=M(1);
	%training with optimum c value for each label
	M=train(lLabels,sparse(lData),sprintf('-s 0 -c %d',c_opt));
        modelSet{l} = M;
        fprintf(fId, '%d Model trained\n', l);
	%predict probability distributions on whole data and write them to intermediate model files
	[predict_label, accuracy, prob_estimates] = predict(lLabels,sparse(lData),M, '-b 1');
        %dlmwrite([Folder, DatasetName, '_labelmodel_', int2str(l), '.y.', int2str(expNum)],prob_estimates, 'delimiter', '\t');
	fprintf(fId, 'Prediction Done for dataset  %s and label %d \n', DatasetName,l);
    	%to create 5 clusters
	cluster=kmeans(prob_estimates(:,1),cluster_count);
	fprintf(fId, 'Clusters created for dataset %s and label %d \n', DatasetName,l);
	for i = 1 : train_instance_count
		for j= 1: N
			annotatorLabels(j,i,l)=getLabelOfAnnotator(j,cluster(i),flipProb,lLabels(i));
		end
	end
	fprintf(fId, 'Annotator labels  created for dataset %s and label %d \n', DatasetName,l);
	end
	for j=1:N
        	dlmwrite([Folder, DatasetName, '_model_', int2str(j), '.y.', int2str(expNum)],squeeze(annotatorLabels(j,:,:)), 'delimiter', '\t');
		fprintf(fId,'written to file for anootator %d\n',j);
	end
end
				
		
