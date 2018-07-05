function [f ,TP,FP,FN,TN] = findFScore2( P, L, beta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% L is the original set of labels
% P is the prediction

TP = sum(sum((L==1) .* (P==1)));
FP = sum(sum((L==-1) .* (P==1)));
FN = sum(sum((L==1) .* (P==-1)));
TN = sum(sum((L==-1) .* (P==-1)));
fprintf('TP is %d\n',TP);
fprintf('FP is %d\n',FP);
fprintf('FN is %d\n',FN);
fprintf('TN is %d\n',TN);
precision = TP/(TP + FP);
recall = TP / (TP + FN);
accuracy=(TP+TN)/(TP+TN+FP+FN);
flippedlabelpercentage=(FP+FN)/(TP+TN+FP+FN);
fprintf('precison is %f\n',precision);
fprintf('recall is %f\n',recall);
fprintf('accuracy is %f\n',accuracy);
fprintf('flipped labels ratio  is %f\n',flippedlabelpercentage);

f = (1 + beta^2)*(precision * recall)/(beta^2 * precision + recall);
if TP == 0
    f = 0;
end

end
