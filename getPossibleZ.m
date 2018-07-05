function [pz]=getPossibleZ(ilms,P)
initialDelta=0.1;
sample_vector=ones(1,1);
nClasses=size(ilms,1);
nModels=size(P,2)/nClasses;
matrixP=reshape(P,[nClasses,nModels]);
matrixP=(matrixP)';
matrixP(matrixP==0)=-1;
probabilities=zeros(nClasses,1);
main_z=zeros(1,nClasses);
for l=1:nClasses
	[predicted_label,~,prob_est]=predict(sample_vector,sparse((matrixP(:,l))'),ilms{l},'-b 1');
	disp('in getPossibleZ printing prob_est');
	fprintf('label %d\n',l);	
	disp(size(prob_est));
	disp(prob_est);
	disp(predicted_label);
	disp(matrixP(:,l)');
	if(predicted_label==1)
		probabilities(l,1)=max(prob_est);
	else
		probabilities(l,1)=1-max(prob_est);
	end
		
end
temp_z=main_z-0.5;
temp_z=abs(temp_z);
[sort_list prob_index]=sort(temp_z);
if(sort_list(3)<=initialDelta)
	initialDelta=sort_list(3);
	problem_index=prob_index(1:3)
elseif((sort_list(3) > initialDelta ) && (sort_list(2) <= initialDelta))
	initialDelta=sort_list(2);
	problem_index=prob_index(1:2);
elseif((sort_list(3) > initialDelta ) && (sort_list(2)  > initialDelta) && (sort_list(1) <=initialDelta))
	initialDelta=sort_list(1);
	problem_index=prob_index(1,1);
else
	
	problem_index=prob_index(1,1);

end
	

main_z(find(probabilities > 0.5+initialDelta))=1;
main_z(find(probabilities < 0.5+initialDelta))=0;

indices_no=size(problem_index,1);
if indices_no==0
	pz=main_z;
else
	for i=0:(2^indices_no)-1
		k=de2bi(i,indices_no);
		main_z(problem_index)=k;
		if(i==0)
			pz=main_z;
		else
			pz=[pz;main_z];	
		end

	end
end

		lId=pz(:,:)==0;
		pz(lId)=-1;
end
