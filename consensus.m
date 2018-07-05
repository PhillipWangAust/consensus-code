num_annotators = 8
instance_count = 500
label_count = 33
annotatorLabels=zeros(num_annotators, instance_count, label_count);

% Read human labels
for i=1:num_annotators
	annotatorLabels(i,:,:)=dlmread(strcat('../scene/user', num2str(i), '.csv'));
end

% Read groundtruth
%labels=dlmread('./scene/groundtruthlabels_500.txt');
%labels=dlmread('../scene/ground_truth.csv');
labels=dlmread('../results/ground_truth_corrected_final.csv');

% Compute labeler-wise kappa
kappa=zeros(num_annotators, 1);
f = zeros(num_annotators, 1);
for i=1:num_annotators
	kappa(i) = findKappaMat_0(squeeze(annotatorLabels(i,:,:)), labels);
	f(i) = findFScore(squeeze(annotatorLabels(i,:,:)), labels, 1);
end
disp(kappa);
disp(f)

% Compute majority consensus
majority = zeros(instance_count, label_count);
temp_m = squeeze(sum(annotatorLabels));
disp(size(temp_m));
majority(temp_m >= 4) = 1;
%disp(temp_m);
kappa_maj = findKappaMat_0(majority, labels);
f_maj = findFScore(majority, labels, 1);
disp(kappa_maj);
disp(f_maj);
dlmwrite('majority.txt', majority);

% Compute MLCM consensus
	m=0;
	%% prediction from each model
	P = zeros(instance_count, label_count * num_annotators);
	for j = 1 : num_annotators
		m=m+1;
		temp_P = squeeze(annotatorLabels(j,:,:)); 
		P(:, (m- 1) * label_count + 1 : m * label_count) = temp_P;
	end

%% MLCM on the prediction
	A = P;
	P(P==0) = -1;
	nModels = num_annotators;
	alpha = 1;

% label matrix for group nodes
	L_machine = eye(label_count);
	B = repmat(L_machine, nModels, 1);
	consensus_MLCM=OnlineCM(A,B,alpha);
%	disp(consensus_MLCM.U);
	L_machine = consensus_MLCM.binarizeProbDist2(consensus_MLCM.U, consensus_MLCM.A);
	L_machine(L_machine==-1) = 0;
	kappa_MLCM = findKappaMat_0(L_machine, labels);
	f_MLCM = findFScore(L_machine, labels, 1);
	disp(kappa_MLCM);
	disp(f_MLCM);
dlmwrite('mlcm.txt', L_machine);

% Compute MLCM-r consensus
%for iter = 1: instance_count
%	m=0;
%% prediction from each model
%	P = zeros(iter, label_count * num_annotators);
%	for j = 1 : num_annotators
%		m=m+1;
%		temp_P = squeeze(annotatorLabels(j,1:iter,:)); 
%		P(:, (m- 1) * label_count + 1 : m * label_count) = temp_P;
%	end

%% MLCM on the prediction
%	A = P;
%	P(P==0) = -1;
%	nModels = num_annotators;
%	alpha = 1;

% label matrix for group nodes
	L_machine = eye(label_count);
	B = repmat(L_machine, nModels, 1);
	[U,Q,K] = TDMLCMr(instance_count, label_count, nModels, A, alpha, B, P);
	L_machine = binarizeProbDist(U, P);
	L_machine(L_machine==-1) = 0;
	kappa_MLCM_r = findKappaMat_0(L_machine, labels);
	f_MLCM_r = findFScore(L_machine, labels, 1);
%end
disp(kappa_MLCM_r);
disp(f_MLCM_r);
dlmwrite('mlcm_r.txt', L_machine);
dlmwrite('q_mat.txt', Q);
