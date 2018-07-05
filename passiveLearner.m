%This script is to test if manual labelllers are getting created properly..
%[train_data train_label test_data test_label]=createManualLabelerswithBias_new(Dataset,N_human,N_exp,fraction,flipProbability);
experiments=3;
fraction=0.7;
flip=0.25;
N_human=6;
Dataset='medical';
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
%create output folders
Folder = '../../Output_new/modelsHuman/';
fId = fopen(strcat(Folder, 'HumanfScore_', Dataset, '.txt'), 'a');
fprintf(fId, '\n-----------------------------------------------------\n');
fprintf(fId,'%s\n',Dataset);
fScore=zeros(experiments,1);
for expNum=1:experiments
	%[train_data train_label test_data test_label]=createManualLabelersWithBias_new(Dataset,N_human,expNum,fraction,0.25);
	fprintf(fId,'expNum is %d\n',expNum);
	%create HumanAnnotator
	ha=HumanAnnotator(Dataset,N_human,fraction,flip);	
        OL = ha.train_label;
	m=0;
        %% prediction from each model
        P = zeros(size(ha.train_data, 1), size(ha.train_label, 2) * (N_human));
        labelCount = size(ha.train_label, 2);
        for j = 1 : N_human
            m=m+1;
            %temp_P = predictLabelsLinear(HumanModel{j}, test_data);
            temp_P = ha.annotator(j,:,:);
	    temp_P=squeeze(temp_P);
            P(:, (m- 1) * labelCount + 1 : m * labelCount) = temp_P;
        end
	fprintf(fId, 'Prediction Done\n');
	%fprintf('Prediction Done\n');
	%% MLCM on the prediction
        index = sum(P, 2) ~= 0;
        index = index .* [1:size(P,1)]';
        index = index(index~=0);
        P = P(index, :);
        OL(OL==0) = -1;
        OL = OL(index, :);
        A = P;
        P(P == 0) = -1;
        nInstances = size(A, 1);
        nClasses = size(ha.train_label, 2);
        nModels = N_human;
        alpha = 1;

        % label matrix for group nodes
        L_human = eye(nClasses);
        B = repmat(L_human, nModels, 1);
        %use OnlineCM over machine labelers for testdata
        %oc_human_testdata=OnlineCM(A,B,alpha);
	fprintf(fId,'using MLCMrClosedForm\n');
	[U Q]=MLCMrClosedForm(nInstances,nClasses,nModels,A,alpha,B);
        fprintf(fId, 'MLCM prediction done\n');
        %fprintf('MLCM prediction done\n');

        %L_human = oc_human_testdata.binarizeProbDist2(oc_human_testdata.U, oc_human_testdata.A);
        L_human = binarizeProbDist(U,P);
        fScore(expNum) = findFScore2(L_human, OL, 1);
        fprintf(fId, 'f score for Consensus labels in expNum %d is %f\n',expNum,fScore(expNum));
        %fprintf('f score in expNum %d is %f\n',expNum,fScore(expNum));
	disp('hello');
	disp(fScore(expNum));
	disp(size(L_human,1));
	fScore_Instance=zeros(experiments,size(L_human,1));

end
