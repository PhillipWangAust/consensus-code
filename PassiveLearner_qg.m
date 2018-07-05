%This script is to run passive learner

%basic variables
N_hum = 3;                     % N_human human models are created
N_mach=1;                    % N_machine models to be created
DatasetName = 'Question-gen';
k = 5;                     % k fold CV is done for training the models
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
Folder = '../../Output_new/modelsMachines/';
Folder1='../../Output_new/modelsHumans/';
Folder2='../../Output_new/';
fId = fopen(strcat(Folder2, 'PassiveLearnerBatch_', DatasetName, '_20.txt'), 'w');
fId2 = fopen(strcat(Folder2, 'PassiveLearnerBatchfScore_', DatasetName, '_20.txt'), 'w');
fId3 = fopen(strcat(Folder2, 'PassiveLearnerBatchconfusion_', DatasetName, '_20.txt'), 'w');
fId4 = fopen(strcat(Folder2, 'PassiveLearnerBatchrunningtime_', DatasetName, '_20.txt'), 'w');
fId5 = fopen(strcat(Folder2, 'PassiveLearnerBatchReliability_', DatasetName, '_20.txt'), 'w');
fprintf(fId, '\n-----------------------------------------------------\n');
fprintf(fId2, '\n-----------------------------------------------------\n');
fprintf(fId, '\nDataset is %s\n',DatasetName);
fprintf(fId2, '\nDataset is %s\n',DatasetName);
fprintf(fId3, '\n-----------------------------------------------------\n');
fprintf(fId3, '\nDataset is %s\n',DatasetName);
fprintf(fId3, '\nDataset is %s\n',DatasetName);
fprintf(fId4, '\nDataset is %s\n',DatasetName);
fprintf(fId4, '\nDataset is %s\n',DatasetName);
fprintf(fId5, '\nDataset is %s\n',DatasetName);
alpha=1;
gamma=0.05;
expNum=1;
test_fraction=0.3;
initialtrainingSize=5;
maxIteration=26;
fScore=zeros(maxIteration,1);
fScore_Ann=zeros(maxIteration,N_mach);
truePositives=zeros(maxIteration,1);
falsePositives=zeros(maxIteration,1);
falseNegatives=zeros(maxIteration,1);
trueNegatives=zeros(maxIteration,1);
machineModelset=cell(N_mach,1);
newmachineModelset=cell(N_mach,1);
tempLabel=cell(N_mach,1);
lLabel=cell(N_mach,1);
labelled_train_instance_count=100;
%get training data and modify it in each experiment.
%disp(size(trainData));

%need to run for some experiments and report the average.
for i=1:expNum
	%initialize the Human Annotatorsi
	fprintf(fId2,'expnum is %d\n',i);
	fprintf(fId3,'expnum is %d\n',i);
	fprintf(fId4,'expnum is %d\n',i);
	fprintf(fId5,'expnum is %d\n',i);
	%disp(size(trainLabel,1));
	Hum_Ann=HumanAnnotator_qg(DatasetName,N_hum,1,0.25);
	%disp(size(trainLabel,1));
	fprintf('before machine models');
	rel=ones(1,N_hum*Hum_Ann.nClasses);
	%ha=initialize(Folder1,DatasetName,N_hum,i);
	P=zeros(labelled_train_instance_count, Hum_Ann.nClasses*Hum_Ann.nModels);       % connection matrix
	i = 0;
	for j=1:Hum_Ann.nModels
		i = i + 1;
		temp_j = squeeze(Hum_Ann.annotator(j,1:labelled_train_instance_count,:));
		P(:,(i-1)*Hum_Ann.nClasses+1:i*Hum_Ann.nClasses) = temp_j;
	end
	disp('In machine Annotator Constructor');
	disp('labelled train instance count');
	disp('P');
	disp(size(P));

	% preprocess the data to remove unpredicted instances 
	d = sum(P~=0, 2);
	%Inst = [1:Hum_Ann.nInstances]';

	%P = P(d~=0, :);         % filter unpredicted rows
	%Inst = Inst(d~=0, :);

	fprintf('total instances in HumanAnnotator is %d\n',Hum_Ann.nInstances);
	nInst = size(P, 1);
	disp(size(P));
	fprintf('ninstaces is %d\n',nInst);

	A = zeros(nInst, Hum_Ann.nClasses * Hum_Ann.nModels);   % Connection matrix with -1 replaced by 0
	A = P;
	%P(P==0) = -1;
	lId = A(:,:) == -1;
	A(lId) = 0;

	%% Run MLCM model
	%Using MLCM closed form
	% label matrix for group nodes
	L = eye(Hum_Ann.nClasses);
	B = repmat(L, Hum_Ann.nModels, 1);
	alpha=1;
	% obtain the consensus probability distribution
	%[U, Q] = MLCMr(nInst, nClasses, nModels, A, alpha, B);

	% use OnlineCM class to get U and Q
	oc_human=OnlineCM(A,B,alpha);
	fprintf('MLCM prediction done\n');
	disp('In machine annotator oc_human.A');
	%disp(oc_human.A);
	L_human= oc_human.binarizeProbDist2(oc_human.U,oc_human.A);
	disp(L_human);
	fprintf('after machine models');
end

