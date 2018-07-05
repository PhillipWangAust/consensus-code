function [P]= getPredictionVector2(P_old,r)
		P=P_old(r,:);
		lId=P(:,:)==-1;
		P(lId)=0;
end
