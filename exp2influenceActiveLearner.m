%This script is to run influenceActiveLearner

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
Folder2='../../Output_new/experiment2/';
initialtrainingSize=50;
alpha=1;
%expNum=1;
test_fraction=0.3;
batch_array=[2,5,10,25,50,100];
%batch_array=[50,100];
machineModelset=cell(N_mach,1);
newmachineModelset=cell(N_mach,1);
tempLabel=cell(N_mach,1);
lLabel=cell(N_mach,1);
%initialize the Human Annotatorsi
%disp(size(trainLabel,1));
ha=HumanAnnotator(DatasetName,N_hum,1,0.25);
%disp(size(trainLabel,1));
fprintf('before machine models');
%ha=initialize(Folder1,DatasetName,N_hum,i);
ma=MachineAnnotator(ha,N_mach,test_fraction,initialtrainingSize);
fprintf('after machine models');
oc_human=ma.oc_obj;

for bat=1:size(batch_array,2)
	batch_size=batch_array(1,bat);
	maxIteration=floor((350-initialtrainingSize)/batch_size)+1;
	fScore=zeros(maxIteration,1);
	truePositives=zeros(maxIteration,1);
	falsePositives=zeros(maxIteration,1);
	falseNegatives=zeros(maxIteration,1);
	trueNegatives=zeros(maxIteration,1);
	fScore_Ann=zeros(maxIteration,N_mach);
%disp(size(trainLabel,1));
%while(initialtrainingSize<=300)
	fId = fopen(strcat(Folder2, 'influencebatch_', DatasetName,'_',num2str(batch_size), '.txt'), 'w');
	fId2 = fopen(strcat(Folder2, 'influencebatchfScore_', DatasetName,'_',num2str(batch_size),'.txt'), 'w');
	fId3 = fopen(strcat(Folder2, 'influencebatchconfusion_', DatasetName,'_',num2str(batch_size),'.txt'), 'w');
	fprintf(fId, '\n-----------------------------------------------------\n');
	fprintf(fId2, '\n-----------------------------------------------------\n');
	fprintf(fId, '\nDataset is %s\n',DatasetName);
	fprintf(fId2, '\nDataset is %s\n',DatasetName);
	%save initial labelled_train_data
	initial_label_data=ma.labelled_train_data;

%get training data and modify it in each experiment.
%disp(size(trainData));

%need to run for some experiments and report the average.
%for i=1:expNum
	for j=1:N_mach
        	temp_machine= squeeze(ma.annotator(j,:,:));
        	lLabel{j}=temp_machine;
        	machineModelset{j}=trainModelLinear(ma.labelled_train_data,lLabel{j});
        	fprintf(fId,'%d Machine Model trained\n',j);
    	end
	%creating models for each label for influence based sampling
	nlabelModels=size(ma.test_label,2);
	influencelabelModelSet=cell(nlabelModels,1);
	newinfluencelabelModelSet=cell(nlabelModels,1);
	featureSet=cell(nlabelModels,1);
	newfeatureSet=cell(nlabelModels,1);
	for labelModel=1:nlabelModels
		temp_labelModel=squeeze(ma.annotator(:,:,labelModel));
		[influencelabelModelSet{labelModel}, featureSet{labelModel}]=trainLabels(temp_labelModel,ma.labelled_train_label(:,labelModel));	
	
	end
		
	
	tempTrainData=ma.labelled_train_data;
	for j=1:N_mach
                tempLabel{j}=lLabel{j};
        end
	tempInfluenceLabel=ma.labelled_train_label;
	%disp(unlabelledCount);	
	% run iterations to compute f score over testdata by adding one instance from unlabelled to labeled data
	for it=1:maxIteration
        	OL = ma.test_label;
		instancesAdded=0;
		m=0;
   		%% prediction from each model
        	P = zeros(size(ma.test_data, 1), size(ma.test_label, 2) * (ma.nModels));
		P_labelled=zeros(size(ma.labelled_train_data,1),size(ma.test_label,2)*(ma.nModels));
		P_unlabelled=zeros(size(ma.unlabelled_train_data,1),size(ma.test_label,2)*(ma.nModels));
        	labelCount = size(ma.test_label, 2);
        	for j = 1 : ma.nModels
			m=m+1;
          		temp_P = predictLabelsLinear(machineModelset{j}, ma.test_data);
	    		fScore_Ann(it,j)=computefScore(temp_P,OL);
	    		%accuracy_Ann(expNum,j)=computeAccuracy(temp_P,OL);
            		fprintf(fId, 'f score for annotator %d in iteration %d is %f\n',j,it,fScore_Ann(it,j));
          		P(:, (m- 1) * labelCount + 1 : m * labelCount) = temp_P;
			P_labelled(:, (m - 1) * labelCount + 1 : m* labelCount) = predictLabelsLinear(machineModelset{j}, ma.labelled_train_data);
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


	% predictions for labelled data
		index_labelled = sum(P_labelled, 2) ~= 0;
                index_labelled = index_labelled .* [1:size(P_labelled,1)]';
                index_labelled = index_labelled(index_labelled~=0);
                P_labelled = P_labelled(index_labelled, :);
                A_labelled = P_labelled;
                P_labelled(P_labelled == 0) = -1;
		lId_ul=A_labelled(:,:)==-1;
		A_labelled(lId_ul)=0;
                nInstances_labelled = size(A_labelled, 1);
                nClasses_labelled = size(ma.test_label, 2);
                nModels = ma.nModels;
                alpha = 1;
		L_machinelabelled = eye(nClasses_labelled);
                B_labelled = repmat(L_machinelabelled, ma.nModels, 1);

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
		%% try getting a instance with least Kappa--as this is uncertain active learning
		%find Consensus over unlabelled Data
                oc_mach_labelled=OnlineCM(A_labelled,B_labelled,alpha);
                %L_machineunlabelled = oc_mach_unlabelled.binarizeProbDist2(oc_mach_unlabelled.U, oc_mach_unlabelled.A);
		%disp(size(selectedInstanceList));
                %disp(unlabelledCount);
		instanceMatrix=[1:size(ha.train_label,1)]';
		unlabelled_index=instanceMatrix(xor(ma.temp_labelled,ma.temp));
		unlabelledCount=size(unlabelled_index,1);
		% Need to calculate expected improvement in Agreement for all unlabelled instances
		Improvement=zeros(unlabelledCount,1);
		predictionVector=cell(unlabelledCount,1);
		indexAdded=zeros(batch_size,1);
                for r = 1 : unlabelledCount
			
        		predictionVector{r} = zeros(1, size(ma.test_label, 2) * (ma.nModels));
			predictionVector{r}=getPredictionVector2(P_unlabelled,r);
			
			%predictionVector{r}=getPredictionVector(machineModelset,ha,ma,unlabelled_index(r,1));
			possibleZ=getPossibleZ(influencelabelModelSet,predictionVector{r});
			imp=0;
			%need to ask Pankaj--problem with getImprovement in Expected Agreement
			for pz=1:size(possibleZ,1)
				calProb=getProbZ(possibleZ(pz,:),influencelabelModelSet,predictionVector{r});
				fprintf(fId,'probability for %d in possibleZ is %f\n',pz,calProb);
				imp=imp+oc_mach_labelled.getImprovementInExpectedAgreement(predictionVector{r},possibleZ(pz,:),@findKappa)*calProb;
			end
			Improvement(r)=imp;
                end
		fprintf(fId,'In iteration %d\n',it);
		for count=1:batch_size
			[R maxIndex]=max(Improvement);
			instanceId=unlabelled_index(maxIndex,1);
			indexAdded(count)=maxIndex;
	%	%	for v=1:ma.nModels
	%			[kap_user temp]=findKappaVec(P_unlabelled(minIndex,(v-1)*nClasses_unlabelled +1 : v* nClasses_unlabelled),L_machineunlabelled(minIndex,:));
	%			fprintf(fId,'Kappa of user %d with consensus over instance with global instance id %d is %f\n',v,instanceId,kap_user);
	%			
	%		end
	%		fprintf(fId,'Agreement over instance with global instance id %d is %f\n',instanceId,R);
			%Kap(minIndex,1)=Inf;	
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
			Improvement(maxIndex,1)=-Inf;
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
                        tempInfluenceLabel=[tempInfluenceLabel;selectedInstanceLabels(1,:)];
			instancesAdded=instancesAdded+1;
		end
		ma.unlabelled_train_data=ha.train_data(xor(ma.temp_labelled,ma.temp),:);
		ma.labelled_train_data=ha.train_data(and(ma.temp_labelled,ma.temp),:);
		fprintf(fId,'no of instances added in iteration %d is %d\n',it,instancesAdded);
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
				fprintf('started traiining label %d\n',l);
    				model = train(lLabs,sparse(lDat),'-s 0 -c 1 -i oldmodel -q');   % train the model on the entire data
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
		%need to train influence based label models as well or not..
		for l=1:nlabelModels
			oldiModel=influencelabelModelSet{l};
			newfeatureSet{l}=getFeatures(featureSet{l},predictionVector,indexAdded,l);
			iDat=newfeatureSet{l};
			%disp('iDat');
			%disp(size(iDat));
			iLabs=tempInfluenceLabel(:,l);
			%disp('iLabs');
			%disp(size(iLabs));
			[oversampleData oversampleLabels]=oversample2(iDat,iLabs);
    			iDat = oversampleData;
    			iLabs = oversampleLabels;	
			newinfluencelabelModelSet{l}=train(iLabs,sparse(iDat),'-s 0 -c 1 -i oldiModel -q');
			fprintf(fId,'Influence training done for label %d\n',l);
		end
		
			
		for l=1:nlabelModels
			influencelabelModelSet{l}=newinfluencelabelModelSet{l};
			featureSet{l}=newfeatureSet{l};
		end
		
		fprintf(fId,'updated influence models\n');
%iteration ends

end

	%initialtrainingSize=initialtrainingSize+step;
	%update training data to previous stage again.
	ma.labelled_train_data=initial_label_data;
	ma.temp_labelled=ma.init_temp_labelled;
	ma.unlabelled_train_data=ha.train_data(xor(ma.temp_labelled,ma.temp),:);
	ma.labelled_train_data=ha.train_data(and(ma.temp_labelled,ma.temp),:);

end
