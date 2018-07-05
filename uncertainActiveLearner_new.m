%This script is to run uncertain ActiveLearner

%basic variables
N_hum = 6;                     % N_human human models are created
N_mach=6;                    % N_machine models to be created
DatasetName = 'enron';
k = 5;                     % k fold CV is done for training the models
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
Folder = '../../Output_new/modelsMachines/';
Folder1='../../Output_new/modelsHumans/';
Folder2='../../Output_new/';
fId = fopen(strcat(Folder2, 'uncertainActiveLearnerBatch_', DatasetName, '_20.txt'), 'w');
fId2 = fopen(strcat(Folder2, 'uncertainActiveLearnerBatchfScore_', DatasetName, '_20.txt'), 'w');
fId3 = fopen(strcat(Folder2, 'uncertainActiveLearnerBatchconfusion_', DatasetName, '_20.txt'), 'w');
fId4 = fopen(strcat(Folder2, 'uncertainActiveBatchrunningtime_', DatasetName, '_20.txt'), 'w');
fprintf(fId, '\n-----------------------------------------------------\n');
fprintf(fId2, '\n-----------------------------------------------------\n');
fprintf(fId, '\nDataset is %s\n',DatasetName);
fprintf(fId2, '\nDataset is %s\n',DatasetName);
fprintf(fId3, '\n-----------------------------------------------------\n');
fprintf(fId3, '\nDataset is %s\n',DatasetName);
fprintf(fId3, '\nDataset is %s\n',DatasetName);
fprintf(fId4, '\nDataset is %s\n',DatasetName);
fprintf(fId4, '\nDataset is %s\n',DatasetName);
alpha=1;
expNum=3;
test_fraction=0.3;
initialtrainingSize=20;
maxIteration=200;
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



%get training data and modify it in each experiment.
%disp(size(trainData));

%need to run for some experiments and report the average.
for i=1:expNum
	%initialize the Human Annotatorsi
	fprintf(fId2,'expnum is %d\n',i);
	fprintf(fId3,'expnum is %d\n',i);
	fprintf(fId4,'expnum is %d\n',i);
	%disp(size(trainLabel,1));
	ha=HumanAnnotator(DatasetName,N_hum,1,0.25);
	%disp(size(trainLabel,1));
	fprintf('before machine models');
	%ha=initialize(Folder1,DatasetName,N_hum,i);
	ma=MachineAnnotator(ha,N_mach,test_fraction,initialtrainingSize);
	fprintf('after machine models');
	oc_human=ma.oc_obj;
	%disp(size(trainLabel,1));
	for j=1:N_mach
        	temp_machine= squeeze(ma.annotator(j,:,:));
        	lLabel{j}=temp_machine;
        	machineModelset{j}=trainModelLinear(ma.labelled_train_data,lLabel{j});
        	fprintf(fId,'%d Machine Model trained\n',j);
    	end
	%disp(size(trainLabel,1));
	%maintain binary matrix of already selected instances
	%selectedInstanceList=zeros(size(trainLabel,1),1);
	%selectedInstanceList(1:200,1)=1;
	%train_inst=size(trainLabel,1);
	%disp(size(trainLabel,1));
	%unlabelledData=trainData(201:train_inst,:);
	%unlabelledCount=size(unlabelledData,1);
	tempTrainData=ma.labelled_train_data;
	for j=1:N_mach
                tempLabel{j}=lLabel{j};
        end
	%disp(unlabelledCount);	
	% run iterations to compute f score over testdata by adding one instance from unlabelled to labeled data
	for it=1:maxIteration
		it_time=tic;
		OL=ma.test_label;
		instancesAdded=0;
		m=0;
   		%% prediction from each model
        	P = zeros(size(ma.test_data, 1), size(ma.test_label, 2) * (ma.nModels));
		P_unlabelled=zeros(size(ma.unlabelled_train_data,1),size(ma.test_label,2)*(ma.nModels));
        	labelCount = size(ma.test_label, 2);
        	for j = 1 : ma.nModels
			m=m+1;
          		temp_P = predictLabelsLinear(machineModelset{j}, ma.test_data);
			%disp('hello123');
			%disp('model and iteration folloew');
			%disp(j);
			%disp(i);
			%disp(machineModelset{j});
			%disp('hello');
			%disp(size(temp_P));
			%disp(size(P));
	    		fScore_Ann(it,j)=computefScore(temp_P,OL);
	    		%accuracy_Ann(expNum,j)=computeAccuracy(temp_P,OL);
            		fprintf(fId, 'f score for annotator %d in iteration %d is %f\n',j,it,fScore_Ann(it,j));
          		P(:, (m- 1) * labelCount + 1 : m * labelCount) = temp_P;
			P_unlabelled(:, (m - 1) * labelCount + 1 : m* labelCount) = predictLabelsLinear(machineModelset{j}, ma.unlabelled_train_data);

        	end

        	fprintf(fId, 'Prediction Done\n');

    	%% MLCM on the prediction
        	index = sum(P, 2) ~= 0;
        	index = index .* [1:size(P,1)]';
        	index = index(index~=0);
        	P = P(index, :);
        	OL = ma.test_label;
        	OL(OL==0) = -1;
        	OL = OL(index, :);
		%disp('size of testlabel');
		%disp(size(ma.test_label));
		%disp(size(OL));
        	A = P;
        	P(P == 0) = -1;
		lId=A(:,:)==-1;
		A(lId)=0;
        	nInstances = size(A, 1);
		%disp('nInstances from A');
		%disp(size(A,1));
		%disp('nInstaces from OL');
		%disp(size(OL,1));
        	nClasses = size(ma.test_label, 2);
        	nModels = ma.nModels;
        	alpha = 1;

    	% label matrix for group nodes
        	L_machine = eye(nClasses);
        	B = repmat(L_machine, nModels, 1);


	% predictions for unlabelled data
		index_unlabelled = sum(P_unlabelled, 2) ~= 0;
                index_unlabelled = index_unlabelled .* [1:size(P_unlabelled,1)]';
                index_unlabelled = index_unlabelled(index_unlabelled~=0);
                P_unlabelled = P_unlabelled(index_unlabelled, :);
                A_unlabelled = P_unlabelled;
                P_unlabelled(P_unlabelled == 0) = -1;
		lId_ul=A_unlabelled(:,:)==-1;
		A_unlabelled(lId_ul)=0;
                nInstances_unlabelled = size(A_unlabelled, 1);
                nClasses_unlabelled = size(ma.test_label, 2);
                nModels = ma.nModels;
                alpha = 1;
		L_machineunlabelled = eye(nClasses_unlabelled);
                B_unlabelled = repmat(L_machineunlabelled, ma.nModels, 1);

        %use OnlineCM over machine labelers for testdata
		oc_mach_testdata=OnlineCM(A,B,alpha);
        	fprintf(fId, 'MLCM prediction done\n');
		if(sum(sum(oc_mach_testdata.U))==0)
			%disp('OnlineCM failed with zero U');
		end
		if(sum(sum(oc_mach_testdata.Q))==0)
			%disp('OnlineCM failed with zero Q');
			%disp('A matrix');
			%disp(oc_mach_testdata.A);
		end
		
		%disp('oc_mach_tetsdata.U');
		%disp(oc_mach_testdata.U);	
		%disp('oc_mach_tetsdata.A');
		%disp(oc_mach_testdata.A);	
        	L_machine = oc_mach_testdata.binarizeProbDist2(oc_mach_testdata.U, oc_mach_testdata.A);
		%disp('L_machine');
		%disp(L_machine);	
		[fScore(it),truePositives(it),falsePositives(it),falseNegatives(it),trueNegatives(it)] = findFScore2(L_machine, OL, 1);
        	fprintf(fId, 'f score in iteration %d is %f\n',it,fScore(it));
		fprintf(fId2,'%d \t %f\n',it,fScore(it));
		fprintf(fId3,'%d \t %d \t %d \t %d \t %d \n',it,truePositives(it),falsePositives(it),falseNegatives(it),trueNegatives(it));
		%% get a random sample from unlabelled data --as this is passive learing
		%% try getting a instance with least Kappa--as this is uncertain active learning
		%find Consensus over unlabelled Data
                oc_mach_unlabelled=OnlineCM(A_unlabelled,B_unlabelled,alpha);
                L_machineunlabelled = oc_mach_unlabelled.binarizeProbDist2(oc_mach_unlabelled.U, oc_mach_unlabelled.A);
		%disp(size(selectedInstanceList));
                %disp(unlabelledCount);
		instanceMatrix=[1:size(ha.train_label,1)]';
		unlabelled_index=instanceMatrix(xor(ma.temp_labelled,ma.temp));
		unlabelled_index=unlabelled_index(index_unlabelled,:);
		unlabelledCount=size(L_machineunlabelled,1);
		Kap=Inf(unlabelledCount,1);
                for r = 1 : unlabelledCount
			Kap(r) = findAgreement(L_machineunlabelled(r, :), P_unlabelled(r, :));
                end
		fprintf(fId,'In iteration %d\n',it);
		for count=1:5
			[R minIndex]=min(Kap);
			instanceId=unlabelled_index(minIndex,1);
			for v=1:ma.nModels
				[kap_user temp]=findKappaVec(P_unlabelled(minIndex,(v-1)*nClasses_unlabelled +1 : v* nClasses_unlabelled),L_machineunlabelled(minIndex,:));
				fprintf(fId,'Kappa of user %d with consensus over instance with global instance id %d is %f\n',v,instanceId,kap_user);
				
			end
			fprintf(fId,'Agreement over instance with global instance id %d is %f\n',instanceId,R);
			Kap(minIndex,1)=Inf;	
			%instanceId = randsample(unlabelled_index,1);
			%disp(size(selectedInstanceList));
			%disp(unlabelledCount);
			%fprintf(fId,'selectedInstance is %d\n',instanceId);
			%if(ma.temp_labelled(instanceId)==0 && ma.temp(instanceId)==1)
			%	%instanceId = randsample(unlabelledCount, 1);+200;
        		%	disp('selected instanceid is valid');
			%	disp(instanceId);
			%	instancesAdded=instancesAdded+1;
			%else
        		%	disp('selected instanceid is not valid');
			%	disp('instanceId');
			%	disp('need to resample again');
			%	continue;
			%end
			ma.temp_labelled(instanceId)=1;
			fprintf(fId,'selected instanceid is %d\n',instanceId);
        		labelCount=size(ma.test_label,2);
        		selectedInstance=zeros(1,size(ma.labelled_train_data,2));
        		selectedInstance=ha.train_data(instanceId,:);
			selectedInstanceLabels=zeros(1,labelCount);
			L_temp=oc_human.binarizeProbDist2(oc_human.U,oc_human.A);
			incrementalAvector=ha.getVfromX(instanceId,3,L_temp,oc_human.A);
			oc_human.incrementalTrain(incrementalAvector);
			u = oc_human.U(oc_human.nInstance,:);
        		a = oc_human.A(oc_human.nInstance,:);
	       		 %%get binarize value for new instance and append it to L
        		selectedInstanceLabels=oc_human.binarizeProbDist(u,a);
			tempTrainData=[tempTrainData;selectedInstance(1,:)];
			for j=1:N_mach
                                tempLabel{j}=[tempLabel{j};selectedInstanceLabels(1,:)];
                        end
			instancesAdded=instancesAdded+1;
		end
		ma.unlabelled_train_data=ha.train_data(xor(ma.temp_labelled,ma.temp),:);
		fprintf(fId,'no of instances added in iteration %d is %d\n',it,instancesAdded);
		%	disp('oc_human matrix sizes');
		%	disp(size(oc_human.A));
		%	disp(size(oc_human.Q));
		%	disp(size(oc_human.DnInv));
			if(rem(it,10)==0)
				tempU=oc_human.getTrueOutput();
				oc_human.U=tempU;
				updateLabels=oc_human.binarizeProbDist2(oc_human.U,oc_human.A);
			
			for j=1:N_mach
				newtemp=tempLabel{j};
				newtemp(end-it*5:end,:)=updateLabels(end-it*5:end,:);
				tempLabel{j}=newtemp;
			end
		end
			
	
		%disp('tempLabel');
		%disp(tempLabel);
        	for j = 1 : N_mach
            		newmachinetrainData = tempTrainData;
            		% consensus output is given as GT
            		newlLabel=tempLabel{j};
    			oldModelMatrix=machineModelset{j};
			for l = 1 : labelCount
    				lDat = newmachinetrainData;
    				lLabs = newlLabel(:, l);
				[oversampleData oversampleLabels]=oversample2(lDat,lLabs);
    				lDat = oversampleData;
    				lLabs = oversampleLabels;	
				oldmodel=oldModelMatrix(l);
				%fprintf('started traiining label %d\n',l);
				%if(rem(it,10)==0)
					%model=train(lLabs,sparse(lDat),'-s 0 -c 1 -q');
				%else
    					model = train(lLabs,sparse(lDat),'-s 0 -c 1 -i oldmodel -q');   % train the model on the entire data
    				%end
				if l == 1
        				modelMatrix = model;
    				else
        				modelMatrix = [modelMatrix model];  % append each label model to the matrix
    				end
				fprintf('ended traiining label %d\n',l);
			end

			newmachineModelset{j}=modelMatrix;
            		fprintf(fId,'%d new Machine Model trained\n',j);
			%disp(size(modelMatrix));
			%disp(size(newmachineModelset{j}));
        	end
        	machinetrainData=newmachinetrainData;
        	for j=1:N_mach
			%disp('last for');
			%disp(machineModelset{j});
            		machineModelset{j}=newmachineModelset{j};
			%disp(newmachineModelset{j});
        	end
		fprintf(fId4,'%d \t %f\n',it,toc(it_time));
		
    end
end
