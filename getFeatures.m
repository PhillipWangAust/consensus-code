function [featureMatrix]=getFeatures(oldfeatureMatrix,P,instanceIds,labelId)
% here we try to update the feature matrix for influence based learning based on predictions of users for new instances
nInstanceAdded=size(instanceIds,1);
disp('getFeatures');
disp(nInstanceAdded);
disp(instanceIds);
disp(size(oldfeatureMatrix));
featureMatrix=oldfeatureMatrix;
for k=1:nInstanceAdded
	r=instanceIds(k,1);	
	nModels=size(oldfeatureMatrix,2);
	nLabels=size(P{r},2)/nModels;
	matrixP=reshape(P{r},[nLabels,nModels]);
	matrixP=(matrixP)';
	matrixP(matrixP==0)=-1;
	featureVector=matrixP(:,labelId);
	featureVector=featureVector';
	featureMatrix=[featureMatrix;featureVector];
end
disp(size(featureMatrix));
end
