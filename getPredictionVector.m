function [P]= getPredictionVector(machineModels,ha1,ma1,instanceId)
nModels=size(machineModels,1);
instance_data=ha1.train_data(instanceId,:);
P = zeros(1, size(ma1.test_label, 2) * (ma1.nModels));

m=0;
labelCount = size(ma1.test_label, 2);
for j = 1 : nModels
	m=m+1;
        temp_P = predictLabelsLinear(machineModels{j}, instance_data);
        P(1, (m- 1) * labelCount + 1 : m * labelCount) = temp_P;
end

		lId=P(:,:)==-1;
		P(lId)=0;
end
