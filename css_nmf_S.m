function [Stest,objhis,obj_diff]= css_nmf_S(V, H, maxiter,tolvalue,showflag)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function learns the corresponding S given a H for cssNMF.
% It is useful for parameter selection based on cross-validation. For
% example, H could be learned on training set and used to derive S on testing
% set.


% Input:
% 1. V is a m*1 cell including a batch of non-negative symmetric
% matrices of size n*n for factorization, n:# of nodes; m:# of subjects
% 2. H is a n*rdim membership matrix, possibly learned on training set.
% rdim: reduced rank
% 3. maxiter: the maximum times of iterations. Default:100
% 4. tolvalue: iteration ends if obj_diffnew<tolvalue. Default: 1e-16
% 5. showflag: 1 or 0. 1: show figures of convergence. Default:0

% Output:
% S is a rdim*m matrix, where each S(j,i) represents the strength of
% component j for subject i, preserving individual differences.
% objhistory records the objective function value of each iteration
% obj_diff records the difference of obj values between two iterations.


% S is guaranteed to converge to its global minimum regardless of the
% initial value

% by Xuan Li, 2016/11/12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check input arguments
if nargin<2,
    error('Please input Association matrices and Membership matrix.');
end
if nargin<3,
    maxiter=100;
end
if nargin<4,
    tolvalue=1e-16;
end
if nargin<5,
    showflag=0;
end

%% Check data
Nsub=length(V);
rdim=size(H,2);

for i=1:Nsub
    % Check if we have non-negative data
    if min(V{i}(:))<0, error('Negative values in data!'); end
    
    % Dimensions
    [vdim, vdim2] = size(V{i});
    if vdim ~= vdim2, error('Input matrix is not symmetric!'); end
end



%% Create initial matrices
rng('shuffle'); % could be commented for reproducibility

% initialize S
Stest=abs(randn(rdim,Nsub));

% calculate initial obj value
objhis=cal_objval(V,Stest,H);

%% Initialize displays
if showflag,
    figure(1); clf; % this will show the objective function value
    drawnow;
end


%% Start iteration
obj_diff=[];
objdiffnew=1;
iter = 0;
while (iter<maxiter) && (objdiffnew>=tolvalue),
    %% Plot the convergence progress
    if showflag==1
        if iter>=2
            figure(1);
            %obj_diff vs # of iter
            plot(log10(obj_diff(2:end)));
        end
        drawnow;
    end
    
    
    
    % Update iteration count
    iter = iter+1;
    
    % Save old values
    Sold = Stest;
    
    
    %% Update S with multiplicative step
    for i=1:Nsub
        S_i=diag(Stest(:,i));
        S_new(:,i)=diag(S_i.*((H'*V{i}*H)./(H'*H*S_i*H'*H+1e-9)));
        % le-9 to avoid zero in the denominator
    end
    Stest=S_new;
    
    % Calculate obj value
    newobj=cal_objval(V,Stest,H);
    
    %disp(['update S'])  ;
    
    objdiffnew=abs(objhis(end)-newobj);
    obj_diff=[obj_diff objdiffnew];
    objhis = [objhis newobj];
    % disp(['iter',num2str(iter),'=',num2str(objdiffnew)]);
end

%%%%
function objval=cal_objval(V,S,H)
sum_err=0;
Nsub=length(V);
for i=1:Nsub
    S_i=diag(S(:,i));
    sum_err=sum_err+sum(sum((V{i}-H*S_i*H').^2));
end
objval = 0.5*sum_err;