function [prob]=getProbZ(Z,ilms,P)
sample_vector=ones(1,1);
nClasses=size(ilms,1);
nModels=size(P,2)/nClasses;
matrixP=reshape(P,[nClasses,nModels]);
matrixP=(matrixP)';
matrixP(matrixP==0)=-1;
probabilities=zeros(nClasses,1);
prob_z=zeros(nClasses,1);
for l=1:nClasses
	[predicted_label,~,prob_est]=predict(sample_vector,sparse((matrixP(:,l))'),ilms{l},'-b 1');
	if(predicted_label==1)
		probabilities(l,1)=max(prob_est);
	else
		probabilities(l,1)=1-max(prob_est);
	end
	if(Z(l)==1)
		prob_z(l)=probabilities(l)
	else
		prob_z(l)=1-probabilities(l);
	end
end
	prob=prod(prob_z);
	disp('prob_z and prob');
	disp(Z);
	disp(prob_z);
	disp(prob);
	disp('probabilties done');
end
