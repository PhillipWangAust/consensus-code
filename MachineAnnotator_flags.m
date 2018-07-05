classdef MachineAnnotator_flags< handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nModels;
        nClasses;
	nInstances;
        %3d matrix which stores database to mimic annotator
        annotator;
        %# user x l matrix and it contains reliability score of user for the label
        %userLabelConfidence;
        %stores sampled user,label pair
	labelled_train_data;
	labelled_train_label;
	labelled_train_index;
	unlabelled_train_data;
	temp;
	temp_labelled;
	init_temp_labelled;
	test_data;
	test_label;
	oc_obj;
	humanLabels;
    end
    
    methods
	function obj = MachineAnnotator_flags(Hum_Ann,nMod,test_fraction,init_mach_train_size)
		% load data in annotator
            	% set number of class, model etc
	   	%disp('called constructor');
	    	%disp(nMod);
	    	obj.nModels=nMod;
		obj.nClasses=Hum_Ann.nClasses;
		data=Hum_Ann.train_data;
		truthLabel=Hum_Ann.train_label;
		total_label_count = Hum_Ann.nClasses;
		total_instance_count = Hum_Ann.nInstances;
		train_label_count=total_label_count;
		test_instance_count=test_fraction*total_instance_count;
		test_instance_count=uint64(test_instance_count);
		train_instance_count=total_instance_count-test_instance_count;
		train_index=randsample(total_instance_count,train_instance_count);
		train_data=data(train_index,:);
		labelled_train_instance_count=init_mach_train_size;
		labelled_train_index=randsample(train_index,labelled_train_instance_count);
		labelled_train_data=data(labelled_train_index,:);
		temp=zeros(total_instance_count,1);
		disp('initial temp');
		disp(size(temp));
		temp(train_index)=1;
		disp(size(temp));
		test_data=data(temp==0,:);
		test_label=truthLabel(temp==0,:);
		temp_labelled=zeros(total_instance_count,1);
		temp_labelled(labelled_train_index)=1;
		unlabelled_train_data=data(xor(temp,temp_labelled),:);
		%now obtain labels from Human Annotators by consensus of their predictions for labelled trained data
		%create labelled and unlabelled partitions
	   	 %disp(Folder);
	   	 %% loading the data from the dataset
		%create subset of Labels for each model
		P=zeros(labelled_train_instance_count, Hum_Ann.nClasses*Hum_Ann.nModels);       % connection matrix
		i = 0;
        	for j=1:Hum_Ann.nModels
                	i = i + 1;
                	temp_j = squeeze(Hum_Ann.annotator(j,labelled_train_index,:));
                	P(:,(i-1)*Hum_Ann.nClasses+1:i*Hum_Ann.nClasses) = temp_j;
            	end
		disp('In machine Annotator Constructor');
		disp('labelled train instance count');
		disp(labelled_train_instance_count);
		disp(init_mach_train_size);
		disp('P');
		disp(size(P));

    		% preprocess the data to remove unpredicted instances 
        	d = sum(P~=0, 2);
        	%Inst = [1:Hum_Ann.nInstances]';

        	P = P(d~=0, :);         % filter unpredicted rows
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
		%disp('In machine Annotator');
		%disp('L_human');
		%disp(L_human);	
		labels=L_human;
		labelled_train_label=labels;
		disp(size(labels))
		labelled_train_index=labelled_train_index(d~=0,:);
		labelled_train_data=data(labelled_train_index,:);
		temp_labelled(labelled_train_index)=1;
		disp('temps');
		disp(size(labelled_train_data));
		disp(size(temp));
		unlabelled_train_data=data(xor(temp,temp_labelled),:);
		labelled_train_instance_count=size(labelled_train_index,1);
		disp(labelled_train_instance_count);
		disp(train_label_count);
		%labels(labels==-1)=0;

		%% Partition data for training and testing
		%data = [trainData;testData];
		%labels = [trainLabel;testLabel];
		%create a partition of training and testing data
		
		%unlabelled_train_label=train_label(temp_labelled==0,:);
		
		%fprintf('total_instances are %d\n',total_instance_count);
		%fprintf('training_instances are %d\n',train_instance_count);
		annotatorLabels=zeros(obj.nModels,labelled_train_instance_count,train_label_count);
		annotatorLabels(1,:,:)=labels;
		disp(size(squeeze(annotatorLabels(1,:,:))))
		%modelSet = cell(train_label_count, 1);
		%fprintf('\n-----------------------------------------------------\n');
		%cluster_count=5;
		%centroid=cell(train_label_count,1);
        	% create a model for each label
		%for l = 1 : train_label_count
        		%lData = labelled_train_data;
        		%lLabels =labelled_train_label(:, l);
			%[oversampleData oversampleLabels]=oversample2(lData,lLabels);
			%disp('displaying oversampled things');
			%disp(size(oversampleData));
			%disp(size(oversampleLabels));
        		%train a model for each label using liblinear
         		%to train using crossfold validation..one time job
        		%M=train(oversampleLabels,sparse(oversampleData),'-s 0 -C -v 5 -q');
        		%c_opt=M(1);
        		%training with optimum c value for each label
        		%M=train(oversampleLabels,sparse(oversampleData),sprintf('-s 0 -c %d -q',c_opt));
        		%modelSet{l} = M;
        		%fprintf('%d Model trained\n', l);
        		%predict probability distributions on whole data and write them to intermediate model files
        		%[predict_label, accuracy, prob_estimates] = predict(lLabels,sparse(lData),M, '-b 1');
        		%fprintf('Prediction Done for dataset  %s and label %d \n', Datasetname,l);
        		%to create 5 clusters
			%disp('displaying prob_estimates');
			%disp(prob_estimates);
			%disp('2nd column only');
			%disp(prob_estimates(:,2));
			%disp(' maximums of probability estimates of class 0');
			%disp(max(prob_estimates(:,2)));
			%max_prob_estimates=sort(prob_estimates,2,'descend');
        		%[cluster,temp_centroid]=kmeans(max_prob_estimates(:,1),cluster_count);
        		%fprintf('Clusters created for dataset %s and label %d \n', Datasetname,l);
			%centroid{l}=temp_centroid;	
        		%for i = 1 : labelled_train_instance_count
                		%for j= 1: obj.nModels
                        	%	annotatorLabels(j,i,l)=getLabelOfMachineAnnotator(j,cluster(i),1-temp_centroid(cluster(i)),lLabels(i));
                		%end
        		%end
        		%fprintf('Annotator labels  created for dataset %s and label %d \n', Datasetname,l);
		%celldisp(centroid);
	    	obj.nClasses=train_label_count;
            	obj.nInstances=labelled_train_instance_count;
		obj.labelled_train_data=labelled_train_data;
		obj.labelled_train_label=labelled_train_label;
		obj.unlabelled_train_data=unlabelled_train_data;
		obj.test_data=test_data;
		obj.test_label=test_label;
		obj.annotator=annotatorLabels;
		obj.oc_obj=oc_human;
		obj.labelled_train_index=labelled_train_index;
		obj.temp_labelled=temp_labelled;
		obj.init_temp_labelled=temp_labelled;
		obj.temp=temp;
		obj.humanLabels=labels;
	     	%obj.userLabelConfidence=zeros(obj.nModels,obj.nClasses);
	     	%disp('Human Annotator created successfully');	    
        end
	function obj = update(obj,Hum_Ann,step)
		data=Hum_Ann.train_data;
		obj.temp_labelled=obj.init_temp_labelled;
		instanceMatrix=[1:size(Hum_Ann.train_label,1)]';
		unlabelled_index=instanceMatrix(xor(obj.temp_labelled,obj.temp));
		newadded_index= randsample(unlabelled_index,step);
		%labels(labels==-1)=0;
		
		%create labelled and unlabelled partitions
		labelled_train_instance_count=size(obj.labelled_train_data,1)+step;
		newadded_train_data=data(newadded_index,:);
		labelled_train_index=[labelled_train_index;newadded_index];
		labelled_train_data=[obj.labelled_train_data;newadded_train_data];
		temp_labelled=obj.init_temp_labelled;
		temp_labelled(newadded_index)=1;
		unlabelled_train_data=data(xor(obj.temp,temp_labelled),:);
		train_label_count=obj.nClasses;
		P=zeros(labelled_train_instance_count, Hum_Ann.nClasses*Hum_Ann.nModels);       % connection matrix
		i = 0;
        	for j=1:Hum_Ann.nModels
                	i = i + 1;
                	temp = squeeze(Hum_Ann.annotator(j,labelled_train_index,:));
                	P(:,(i-1)*Hum_Ann.nClasses+1:i*Hum_Ann.nClasses) = temp;
            	end

    		% preprocess the data to remove unpredicted instances 
        	d = sum(P~=0, 2);
        	%Inst = [1:Hum_Ann.nInstances]';

        	P = P(d~=0, :);         % filter unpredicted rows
        	%Inst = Inst(d~=0, :);

		fprintf('total instances in HumanAnnotator is %d\n',Hum_Ann.nInstances);
        	nInst = size(P, 1);
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
		labelled_train_label=L_human;
		
		%again do consensus for newly added data to get their labels.
		%fprintf('training_instances are %d\n',train_instance_count);
		annotatorLabels=zeros(obj.nModels,labelled_train_instance_count,obj.nClasses);
		modelSet = cell(train_label_count, 1);
		%fprintf('\n-----------------------------------------------------\n');
		cluster_count=5;
		centroid=cell(train_label_count,1);
        	% create a model for each label
		for l = 1 : obj.nClasses
        		lData = labelled_train_data;
        		lLabels =labelled_train_label(:, l);
			[oversampleData oversampleLabels]=oversample2(lData,lLabels);
			%disp('displaying oversampled things');
			%disp(size(oversampleData));
			%disp(size(oversampleLabels));
        		%train a model for each label using liblinear
         		%to train using crossfold validation..one time job
        		M=train(oversampleLabels,sparse(oversampleData),'-s 0 -C -v 5 -q');
        		c_opt=M(1);
        		%training with optimum c value for each label
        		M=train(oversampleLabels,sparse(oversampleData),sprintf('-s 0 -c %d -q',c_opt));
        		modelSet{l} = M;
        		%fprintf('%d Model trained\n', l);
        		%predict probability distributions on whole data and write them to intermediate model files
        		[predict_label, accuracy, prob_estimates] = predict(lLabels,sparse(lData),M, '-b 1');
        		%fprintf('Prediction Done for dataset  %s and label %d \n', Datasetname,l);
        		%to create 5 clusters
			%disp('displaying prob_estimates');
			%disp(prob_estimates);
			%disp('2nd column only');
			%disp(prob_estimates(:,2));
			%disp(' maximums of probability estimates of class 0');
			%disp(max(prob_estimates(:,2)));
			max_prob_estimates=sort(prob_estimates,2,'descend');
        		[cluster,temp_centroid]=kmeans(max_prob_estimates(:,1),cluster_count);
        		%fprintf('Clusters created for dataset %s and label %d \n', Datasetname,l);
			centroid{l}=temp_centroid;	
        		for i = 1 : labelled_train_instance_count
                		for j= 1: obj.nModels
                        		annotatorLabels(j,i,l)=getLabelOfMachineAnnotator(j,cluster(i),1-temp_centroid(cluster(i)),lLabels(i));
                		end
        		end
        		%fprintf('Annotator labels  created for dataset %s and label %d \n', Datasetname,l);
       		end
		%celldisp(centroid);
            	obj.nInstances=labelled_train_instance_count;
		obj.labelled_train_data=labelled_train_data;
		obj.labelled_train_label=labelled_train_label;
		obj.unlabelled_train_data=unlabelled_train_data;
		obj.annotator=annotatorLabels;
		obj.temp_labelled=temp_labelled;
		obj.init_temp_labelled=temp_labelled;
		obj.oc_obj=oc_human;
		obj.humanLabels=labelled_train_label;
		obj.labelled_train_index=labelled_train_index;
		
	     	%obj.userLabelConfidence=zeros(obj.nModels,obj.nClasses);
	     	%disp('Human Annotator created successfully');	    
        end
        
   end
end
