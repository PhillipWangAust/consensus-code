%% This script is to test the relation between cosnensus maximization and ground truth
% We take the prediction of each of the models and plot the consensus
% against the F measure wrt ground truth, taking 1 %, 2 %, 3 %.. from top
% and bottom of the instances.

%% Parameters
N_hum = 6;                     % N_human human models are created
DatasetName = 'corel5k';
k = 5;                     % k fold CV is done for training the models
libLinearPath = '../../../liblinearinc-2.1/matlab';
config.libLinearPath = libLinearPath;
addpath(libLinearPath);
Folder2='../../Output_new/userReduction/';
fId = fopen(strcat(Folder2, 'userreductionLog_', DatasetName, '.txt'), 'w');
fId2 = fopen(strcat(Folder2, 'userReductionfScore_', DatasetName, '.txt'), 'w');
fprintf(fId, '\n-----------------------------------------------------\n');
fprintf(fId2, '\n-----------------------------------------------------\n');
fprintf(fId, '\nDataset is %s\n',DatasetName);
fprintf(fId2, '\nDataset is %s\n',DatasetName);
alpha=1;
%expNum=3;
%test_fraction=0.3;
%initialtrainingSize=1000;
%maxIteration=101;
%fScore=zeros(maxIteration,1);
%fScore_Ann=zeros(maxIteration,N_mach);
%truePositives=zeros(maxIteration,1);
%falsePositives=zeros(maxIteration,1);
%falseNegatives=zeros(maxIteration,1);
%trueNegatives=zeros(maxIteration,1);
%machineModelset=cell(N_mach,1);
%newmachineModelset=cell(N_mach,1);
%tempLabel=cell(N_mach,1);
%lLabel=cell(N_mach,1);
%get training data and modify it in each experiment.
%disp(size(trainData));

%need to run for some experiments and report the average.
for expNum = 1 : 5
	N=N_hum;
    % for each experiment this is repeated
    	fprintf(fId,'\nExperiment Number = %d\n', expNum);
	%initialize the Human Annotatorsi
	fprintf(fId2,'expnum is %d\n',i);
	ha=HumanAnnotator(DatasetName,N_hum,1,0.25);
	%disp(size(trainLabel,1));

    	models = ones(N_hum, 1);
    	%models(1:5) = 0;
   	 modelIndex = [1:N_hum]';
	disp(models);
  	  %modelIndex = modelIndex(models==1);
   	 maxIteration = 2;
 	 capacityMatrix = zeros(N_hum, maxIteration);
    for it = 1 : maxIteration
        %% Read the data
	nInst=ha.nInstances;
	nClasses=ha.nClasses;
        nModels = sum(models);
        P=zeros(nInst, nClasses*nModels);	% connection matrix
        i = 0;
        for j=1:N_hum
		if(models(j)==1)
                	i = i + 1;
        		temp= squeeze(ha.annotator(j,:,:));
                	P(:,(i-1)*nClasses+1:i*nClasses) = predictionConvert(temp);
        	end
	end
	str1=strcat(DatasetName,'_P_',num2str(expNum),'_',num2str(it),'.txt');
	str2=strcat(DatasetName,'_L_',num2str(expNum),'_',num2str(it),'.txt');
        % load ground truth
        OL = ha.train_label;

        % preprocess the data to remove unpredicted instances 
        d = sum(P~=0, 2);
        Inst = [1:nInst]';

        P = P(d~=0, :);         % filter unpredicted rows
	dlmwrite(str1,P);	
        OL = OL(d~=0, :);       % filter unpredicted rows
        Inst = Inst(d~=0, :);
        OL(OL == 0) = -1;
	dlmwrite('TrueLabels.txt',OL);	

        oldInst = nInst;
        nInst = size(P, 1);


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

        % Closed form values of U and Q
	%oc_temp=OnlineCM(A,B,alpha);
	disp(size(A));
	disp(size(B));
        [U, Q] = MLCMrClosedForm(nInst, nClasses, nModels, A, alpha, B);
        fprintf(fId,'MLCM prediction done\n');


        %% Obtain the Labels using threshold
	
        L = binarizeProbDist(U, P);
	dlmwrite(str2,L);	
	
        %L = oc_temp.binarizeProbDist2(oc_temp.U, oc_temp.A);
        fprintf(fId,'Binarization Done\n');
        
        [fScore,TP,FP,FN,TN] = findFScore2(L, OL, 1);
        fprintf(fId,'f score is %f\n', fScore);
        fprintf(fId,'TP is %d\n', TP);
        fprintf(fId,'FP is %d\n', FP);
        fprintf(fId,'FN  is %d\n', FN);
        fprintf(fId,'TN is %d\n', TN);
        fprintf(fId2,'f score is %f\n', fScore);

        userConfidenceMicro = zeros(nModels, 1);
        for i = 1 : nModels
            userConfidenceMicro(i) = findKappaUser(L, P(:, (i - 1) * nClasses + 1 : i * nClasses));
        end

        userConfidenceMacro = zeros(nModels, 1);
        for i = 1 : nModels
            userConfidenceMacro(i) = findUserConfidenceMacro(L, P(:, (i - 1) * nClasses + 1 : i * nClasses));
	    fprintf(fId,'Kappa values for User %d is %f\n',i,userConfidenceMacro(i));
        end

        userConfidenceMean = zeros(nModels, 1);
        LRep = repmat(L, 1, nModels);
        K = findKappaUserLabel(P, LRep);          % kappa values for each user , label
        for i = 1 : nModels
            temp = K((i - 1) * nClasses + 1 : i * nClasses);
            userConfidenceMean(i) = sum(temp) / sum(temp~=-2);
        end

        userCapacity = userConfidenceMacro;
        output = zeros(N, 1);
        output(models==1) = userCapacity;
        capacityMatrix(:, it) = output; 
        
        if it> 1
            change = sum(models.*(capacityMatrix(:, it) - capacityMatrix(:, it - 1)));
            fprintf(fId,'change aggregate is %f\n', change);
        end

        % We remove the bottom k users and see if there is improvement
        k = 1;
        [~, index] = sort(userCapacity);
        modelsRemoved = modelIndex(index(1:k));
        fprintf(fId,'Removed Model = %d\n',modelsRemoved);
        disp(modelsRemoved);
        modelIndex = sort(modelIndex(index(k+1:end)));
        models(modelsRemoved) = 0;
    end
    disp(capacityMatrix);
    
    
    
    %% Random removal of models

    
    for expIteration = 6 : N
        fprintf(fId,'expIteration = %d\n', expIteration);
        maxIteration = 2;
        capacityMatrix = zeros(N, maxIteration);
        models = ones(N, 1);
        %models(1 : 5) = 0;
        modelIndex = [1:N]';
        %modelIndex = modelIndex(models == 1);
        for it = 1 : maxIteration
            %% Read the data
		nInst=ha.nInstances;
		nClasses=ha.nClasses;
		nModels=sum(models);
       		 P=zeros(nInst, nClasses*nModels);	% connection matrix
       	 	i = 0;
        	for j=1:N_hum
            		if models(j) == 1
                		i = i + 1;
        			temp= squeeze(ha.annotator(j,:,:));
                		P(:,(i-1)*nClasses+1:i*nClasses) = predictionConvert(temp);
      		  	end
		end
       	 % load ground truth
        	OL = ha.train_label;
		str1=strcat(DatasetName,'_P_',num2str(expNum),'_',num2str(it),'.txt');
		str2=strcat(DatasetName,'_L_',num2str(expNum),'_',num2str(it),'.txt');


        	% preprocess the data to remove unpredicted instances 
       		 d = sum(P~=0, 2);
        	Inst = [1:nInst]';

        	P = P(d~=0, :);         % filter unpredicted rows
		dlmwrite(str1,P);	
        	OL = OL(d~=0, :);       % filter unpredicted rows
        	Inst = Inst(d~=0, :);
        	OL(OL == 0) = -1;
		dlmwrite('TrueLabels.txt',OL);	

        	oldInst = nInst;
        	nInst = size(P, 1);


        	A = zeros(nInst, nClasses * nModels);   % Connection matrix with -1 replaced by 0
        	A = P;
        	%P(P==0) = -1;
       	 	lId = A(:,:) == -1;
        	A(lId) = 0;

        	%% Run MLCM model
        	% Using MLCM closed form
        	% label 	matrix for group nodes
       		 L = eye(nClasses);
        	B = repmat(L, nModels, 1);

        % obtain the consensus probability distribution
        %[U, Q] = MLCMr(nInst, nClasses, nModels, A, alpha, B);

        % Closed form values of U and Q
       		 [U, Q] = MLCMrClosedForm(nInst, nClasses, nModels, A, alpha, B);
        	fprintf(fId,'MLCM prediction done\n');


        	%% Obtain the Labels using threshold
	
        	L = binarizeProbDist(U, P);
		dlmwrite(str2,L);	
		%oc_temp=OnlineCM(A,B,alpha);
        	%[U, Q] = MLCMrClosedForm(nInst, nClasses, nModels, A, alpha, B);

        	fprintf(fId,'Binarization Done\n');
        
        	[fScore,TP,FP,FN,TN] = findFScore2(L, OL, 1);
        	fprintf(fId,'f score is %f\n', fScore);
       	 	fprintf(fId,'TP is %d\n', TP);
        	fprintf(fId,'FP is %d\n', FP);
        	fprintf(fId,'FN  is %d\n', FN);
        	fprintf(fId,'TN is %d\n', TN);
        	fprintf(fId2,'f score is %f\n', fScore);

        	userConfidenceMicro = zeros(nModels, 1);
        	for i = 1 : nModels
            		userConfidenceMicro(i) = findKappaUser(L, P(:, (i - 1) * nClasses + 1 : i * nClasses));
        	end

        	userConfidenceMacro = zeros(nModels, 1);
        	for i = 1 : nModels
            		userConfidenceMacro(i) = findUserConfidenceMacro(L, P(:, (i - 1) * nClasses + 1 : i * nClasses));
	    	fprintf(fId,'Kappa values for User %d is %f\n',i,userConfidenceMacro(i));
        	end

            userConfidenceMean = zeros(nModels, 1);
            LRep = repmat(L, 1, nModels);
            K = findKappaUserLabel(P, LRep);          % kappa values for each user , label
            for i = 1 : nModels
                temp = K((i - 1) * nClasses + 1 : i * nClasses);
                userConfidenceMean(i) = sum(temp) / sum(temp~=-2);
            end

            userCapacity = userConfidenceMacro;
            output = zeros(N, 1);
            output(models==1) = userCapacity;
            capacityMatrix(:, it) = output; 

            if it> 1
                change = sum(models.*(capacityMatrix(:, it) - capacityMatrix(:, it - 1)));
                fprintf(fId,'change aggregate is %f\n', change);
            end

            % We remove the bottom k users and see if there is improvement
            k = 1;
            index = randperm(size(userCapacity, 1));
            %modelsRemoved = modelIndex(index(1:k));
            modelsRemoved = expIteration;
            fprintf(fId,'Removed Model = %d\n',modelsRemoved);
            disp(modelsRemoved);
            modelIndex = sort(modelIndex(index(k+1:end)));
            models(modelsRemoved) = 0;
        end
    end
    
    
end
%exit;
