
   classdef OnlineCM < handle
  %% write a description of the class here.
% running online CM
%    init = 400;
%    onlineCM = OnlineCM( A(1:init,:),B,alpha);
%    %L is binarized U
%    %binarize U for initial value
%    L1 = onlineCM.binarizeProbDist2(onlineCM.U,onlineCM.A);
%    for i = A(init+1:nInst,:)'
%        onlineCM.incrementalTrain(i');
%        disp(onlineCM.nInstance);
%        u = onlineCM.U(onlineCM.nInstance,:);
%        a = onlineCM.A(onlineCM.nInstance,:);
%        %get binarize value for new instance and append it to L
%        L1(onlineCM.nInstance,:)= onlineCM.binarizeProbDist(u,a);
%%    end



   
   
       properties
       % define the properties of the class here, (like fields of a struct)
           A;
           B;
           alpha;
           nInstance;
           nGroup;
           nModel;
           nClass;
           U;
           Q;
           w;
           wn;
           DvInv;
           DnInv;
           ADA;
           Dl;
           Doml;
           DomlY;
           Iv;
           S;
           temp;
           %L;
       end
       methods
       % methods, including the constructor are defined in this block
       
 
            
      % function obj = OnlineCM()
           % class constructor
           
           %disp('online CM');
       %end
       
       function l = binarizeProbDist(obj,u,a)
           p=a;
           p(p==0) = -1;
           nClasses = size(u, 2);              % Number of labels
            %nModels = size(p, 2) / nClasses;    % Number of models 
            l = u;                              % Predicted values
            
            % find the threshold for the case
            tmp = u;
            [tmp, ~] = sort(tmp);
            threshold = (tmp(1:nClasses-1) + tmp(2:nClasses)) / 2;
            maxK = 0;
            maxTau = 0;
            for tau = [0 threshold 1];
                tmp = u;
                tmp(tmp <= tau) = -1;
                tmp(tmp > tau) = 1;
                avgK = findAgreement(tmp, p);
                if avgK > maxK
                    maxTau = tau;
                    maxK = avgK;
                end
            end
            l( u(:) <= maxTau) = -1;
            l( u(:) > maxTau) = 1;
       end
       
       function L1 = binarizeProbDist2(obj, U1, A1)
           %function doesnt work properly. Gives wrong answer
                L1 = U1;
                A1 = A1;
                nInst1 = size(U1, 1);
                for i1 = 1:nInst1
                    L1(i1,:)= obj.binarizeProbDist(U1(i1,:),A1(i1,:));
                end
       end
       
       function obj = OnlineCM(A,B,alpha)
            obj.A=A;
            obj.B=B;
            obj.alpha=alpha;
            [obj.nGroup, obj.nClass] = size(B);
            [obj.nInstance, obj.nGroup] = size(A);
            obj.nModel = obj.nGroup/obj.nClass;
            obj.U = 1 / obj.nClass * ones(obj.nInstance, obj.nClass);
            obj.Q = zeros(obj.nClass * obj.nModel, obj.nClass);
            obj.w = (sum(A, 1)+0.01);
            obj.wn = (sum(A, 2)+0.01);
            obj.DvInv = diag(1./obj.w);
            obj.DnInv = diag(1./obj.wn);
            obj.ADA = obj.A'*obj.DnInv*obj.A;
            obj.Dl = diag(1./(sum(obj.A, 1) + obj.alpha))*diag(sum(obj.A,1));
            obj.Doml = diag(obj.alpha./(sum(obj.A, 1) + obj.alpha));
            obj.S= obj.DvInv * obj.ADA;
            obj.Iv = eye(obj.nModel * obj.nClass);
            obj.temp =inv(obj.Iv - obj.Dl * obj.S);
            obj.DomlY = obj.Doml * obj.B;
            obj.Q = obj.temp * obj.DomlY;
            obj.U = obj.DnInv * obj.A * obj.Q;
            %obj.L = binarizeProbDist2(obj.U,obj.A);
            %obj.L = obj.U;
       end
           
       function TU = getTrueOutput(obj)
            obj.wn = (sum(obj.A, 2)+0.01);
            obj.DnInv = diag(1./obj.wn);
		
           TU = obj.DnInv * obj.A * obj.Q;
       end
       
       function obj = incrementalTrain(obj,a_i)
            %update A
            obj.A(obj.nInstance+1,:)=a_i;
            %update nInstance
            obj.nInstance = obj.nInstance+1;
            %update w, dvInv, Dl, Doml, DomlY
            j=1;
            for i = a_i
                if i > 0
                    obj.w(j) = obj.w(j) + i;
                    obj.DvInv(j,j) = 1/obj.w(j);
                    obj.Dl(j,j) = obj.w(j)/(obj.w(j)+obj.alpha);
                    obj.Doml(j,j) = obj.alpha/(obj.w(j)+obj.alpha);
                    obj.DomlY(j,:) = obj.Doml(j,j) * obj.B(j,:);
                end
                j=j+1;
            end
            
           obj.wn(obj.nInstance) = sum(a_i)+0.01;

           %update dnInv --no need
            %sparse(1:obj.nInstace,1:obj.iInstance,1./obj.wn)
            
            %update ADA
            for i = find(a_i)
                for j = find(a_i)
                    obj.ADA(i,j) = obj.ADA(i,j) + a_i(i)*a_i(j)/obj.wn(obj.nInstance);
                end
            end
            
            %update S
            
            obj.S= sparse(1:obj.nGroup,1:obj.nGroup,1./obj.w) * obj.ADA;
            
            %update temp
            %can do optimization here
            obj.temp =inv(obj.Iv - obj.Dl * obj.S);
            %update Q
            obj.Q = obj.temp * obj.DomlY;
            %update U in streaming manner that is change only last entry
            %display('size of a_i and obj.Q');
            %disp(size(a_i) );
            %disp(size(obj.Q));
            u1 = a_i * obj.Q;
            u = u1*1/obj.wn(obj.nInstance);
            %u = obj.DnInv(obj.nInstance,obj.nInstance)*u1;
            %disp('instance');
            %disp(a_i);
            %disp(u);

            %disp('count');
            %disp(obj.nInstance);
            %disp('size of u');
            %disp(size(obj.U));
            obj.U(obj.nInstance,:) = u;
            %obj.L(obj.nInstance,:)= binarizeProbDist(u,a_i);
       end
	 %%pass the x as an row of matrix A. It uses equation 13 and 14 of
       %%the writeup. You can use any function to measure aggrement.
       function kdiff = getImprovementInExpectedAgreement(obj,a_i,z,aggrementFuncHandle)
           denom = sum(a_i);
           
           %%ASK: WHAT LOGIC SHOULD I FOLLOW IN CASE OF NO LABEL
           if denom == 0
               denom = 0.01;
           end
           %q1 will have current Q^* while q2 will have updated Q^*
           q1 = obj.Q;
           obj1=obj;
           q2 = obj1.incrementalTrain(a_i).Q;
           
           %use quation 13 to compute u1 and equation 14 to compute u2..
           kdiff = 0;
           diffQ = ((q1 - q2) ~= 0);
           if sum(diffQ(:)) ~= 0
               u1 = a_i*q1/denom;
               u2 = a_i*q2/denom;
               %disp('diff U');
                %disp (u1 - u2);
                %binarize u1 and u2
                l1 = obj.binarizeProbDist(u1,a_i);
                l2 = obj.binarizeProbDist(u2,a_i);
                %compute improvement in expected aggrement
                diffL = ((l1 - l2) ~= 0);
                kdiff = aggrementFuncHandle(l2,z) - aggrementFuncHandle(l1,z);

                if sum(diffL(:)) ~=0
                    disp('Labels has changed and kdiff is ');
                    disp(kdiff);
                    disp('k1 is');
                    disp(aggrementFuncHandle(l1,z));
                    disp('k2 is');
                    disp(aggrementFuncHandle(l2,z));

                end
           end
           
       end
       
   end
end


