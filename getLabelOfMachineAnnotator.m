function [label]=getLabelofMachineAnnotator(annotator,instanceCluster,flipProb,trueLabel)
% this fnction is to get label of annotator for a instance in given cluster
% we use flip probbaility to flip truelabels if annotator and instance cluster are not same with flipProb.
if(flipProb==0)
	label=trueLabel;
elseif(annotator==instanceCluster)
	label=trueLabel;
else
	temp=round(1/flipProb);
	val=randi([1,temp],1,1);
	if(val == 1) 
		label=-trueLabel;	
	else
		label= trueLabel;
	end
end


