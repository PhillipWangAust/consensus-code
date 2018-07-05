function[ovData,ovLabels]=oversample(Data,Labels)
positiveinstances=sum(Labels);
negativeinstances=size(Labels,1)-positiveinstances;
if(positiveinstances > 0)
%	disp('entered if in oversample');
%	disp('positive instances');
%	disp(positiveinstances);
%	disp(negativeinstances);
	ratio=negativeinstances/positiveinstances;
	ratio=uint64(ratio);
	%disp('ratio in if block');
	%disp(ratio);
	positiveindex=find(Labels);
	tobeaddedData=Data(positiveindex,:);
	tobeaddedLabels=Labels(positiveindex,:);
	ovData=Data;
	ovLabels=Labels;
	while(ratio > 1.1)
		%disp('in while loop in oversample');
		ovData=[ovData;tobeaddedData];
		ovLabels=[ovLabels;tobeaddedLabels];
		positiveinstances=sum(ovLabels);
		negativeinstances=size(ovLabels,1)-positiveinstances;
		ratio=negativeinstances/positiveinstances;
		ratio=uint64(ratio);
		%disp('ratio');
		%disp(ratio);
	end
		%disp('exited while loop in oversample');
else
	%disp('entered else in oversample');
	ovData=Data;
	ovLabels=Labels;
end
end
