classdef HumanAnnotator_flags< handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nModels;
        nClasses;
	nInstances;
        %3d matrix which stores database to mimic annotator
        annotator;
        %# user x l matrix and it contains reliability score of user for the label
        userLabelConfidence;
        %stores sampled user,label pair
	train_data;
	train_label;
	test_data;
	test_label;
    end
    
    methods
	function obj = HumanAnnotator_flags(Datasetname,nMod,fraction,flipProbability)
		% load data in annotator
            	% set number of class, model etc
	   	%disp('called constructor');
	    	%disp(nMod);
	    	obj.nModels=nMod;
	   	 %disp(Folder);
	   	 %% loading the data from the dataset
		[trainData, trainLabel, testData, testLabel] = loadDataset(Datasetname);


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
		%fprintf('total_instances are %d\n',total_instance_count);
		%fprintf('training_instances are %d\n',train_instance_count);
		annotatorLabels=zeros(obj.nModels,train_instance_count,train_label_count);
		modelSet = cell(train_label_count, 1);
		flipProb=flipProbability;
		%fprintf('\n-----------------------------------------------------\n');
		%train_index=randsample(total_instance_count,train_instance_count);
		%train_data=data(train_index,:);
		%train_label=labels(train_index,:);
		train_data=data;
		train_label=labels;
		%temp=zeros(total_instance_count,1);
		temp=ones(total_instance_count,1);
		%temp(train_index)=1;
		test_data=data(temp==0,:);
		test_label=labels(temp==0,:);
		%cluster_count=5;
		%centroid=cell(train_label_count,1);
        	% create a model for each label
		%for l = 1 : train_label_count
        		%lData = train_data;
        		%lLabels = train_label(:, l);
			%[oversampleData oversampleLabels]=oversample(lData,lLabels);
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
        		%for i = 1 : train_instance_count
                        		annotatorLabels(1,:,:)=dlmread('./flags_labels/user1.csv');
                        		annotatorLabels(2,:,:)=dlmread('./flags_labels/user2.csv');
                        		annotatorLabels(3,:,:)=dlmread('./flags_labels/user3.csv');
                        		annotatorLabels(4,:,:)=dlmread('./flags_labels/user4.csv');
                        		annotatorLabels(5,:,:)=dlmread('./flags_labels/user5.csv');
                        		annotatorLabels(6,:,:)=dlmread('./flags_labels/user6.csv');
                        		annotatorLabels(7,:,:)=dlmread('./flags_labels/user7.csv');
                        		annotatorLabels(8,:,:)=dlmread('./flags_labels/user8.csv');
        		%end
        		%fprintf('Annotator labels  created for dataset %s and label %d \n', Datasetname,l);
       		%end
		%celldisp(centroid);
	    	obj.nClasses=train_label_count;
            	obj.nInstances=train_instance_count;
		obj.train_data=train_data;
		obj.train_label=train_label;
		obj.test_data=test_data;
		obj.test_label=test_label;
		obj.annotator=annotatorLabels;
	     	obj.userLabelConfidence=zeros(obj.nModels,obj.nClasses);
	     	%disp('Human Annotator created successfully');	    
        end
        
        function obj = computeUserLabelRel(obj,L,A)
            P=A;
            P(A==0)=-1;
            for j=1:obj.nModels %no.of human annotators
                for r=1:obj.nClasses
                    obj.userLabelConfidence(j,r)=findKappaUserLabel(L(:,r),P(:, ((j - 1) * obj.nClasses) + r));
                end
            end
            tmp = sum(obj.userLabelConfidence,1);
            obj.userLabelConfidence = obj.userLabelConfidence*diag(1./tmp)
        end
        
        function M = sampleUserLabelPair(obj,k)
            %it is a cumulative prob
            uLRange = cumsum(obj.userLabelConfidence);
            M=zeros(obj.nClasses*k,2);
            p=1;
            for i = 1:obj.nClasses
                for j = 1:k
                    %Generate Random number r
                    r = rand();
                    ind1 = sum(uLRange(:,i) < r)+1;
                    %ind2 = uLRange(:,i) >= r;
                    % select appropriate user
                    M(p,1) = ind1;
                    M(p,2) = i;
                    p=p+1;
                end
            end
        end
        
        function v = getVfromX(obj,xid,k,L,A)
            v = zeros(1,obj.nClasses*obj.nModels);
            obj.computeUserLabelRel(L,A);
            M = obj.sampleUserLabelPair(k);
            for i = 1:k*obj.nClasses
                u = M(i,1);
                l = M(i,2);
                v(1,(u-1)*obj.nClasses+l)=obj.annotator(u,xid,l);
            end
        end
    end
    
end
