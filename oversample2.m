function[ovData,ovLabels]=oversample2(Data,Labels)
%disp('entered oversample2');
p=find(Labels==1);
positiveinstances=size(p,1);
negativeinstances=size(Labels,1)-positiveinstances;
if(positiveinstances > 0)
%	disp('entered if in oversample2');
%	disp('positive instances');
%	disp(positiveinstances);
%	disp(negativeinstances);
	ratio=negativeinstances/positiveinstances;
	ratio=uint64(ratio);
%	disp('ratio in if block');
%	disp(ratio);
	positiveindex=find(Labels==1);
	tobeaddedData=Data(positiveindex,:);
	tobeaddedLabels=Labels(positiveindex,:);
	ovData=Data;
	ovLabels=Labels;
	while(ratio > 1.1)
		%disp('in while loop in oversample2');
		ovData=[ovData;tobeaddedData];
		ovLabels=[ovLabels;tobeaddedLabels];
		p=find(ovLabels==1);
		positiveinstances=size(p,1);
		negativeinstances=size(ovLabels,1)-positiveinstances;
		ratio=negativeinstances/positiveinstances;
		ratio=uint64(ratio);
		%disp('ratio');
		%disp(ratio);
	end
		%disp('exited while loop in oversample2');
else
	%disp('entered else in oversample2');
	ovData=Data;
	ovLabels=Labels;
end
%disp('leaving oversample2');
end
