function [H,S,objhistory]= css_nmf( V, rdim, b,maxiter,tolvalue, fname, showflag )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%css_nmf: collective sparse symmetric NMF (Tri-factorization)

% This function solves the symmetric NMF problem with sparseness
% constraints for a batch of matrices

% Problem statement (Li et al, 2018):
% min 0.5* \sigma_i=1 to m ||V_i-H*S_i*H'||_2^2+\b*||H||_1  s.t. H,S_i>=0,
% where b is the regularization parameter, and each column of H is
% constrainted to have a unit norm.

% It is written based on the code in Hoyer 2004.

% For solving the diagonal matrix S_i, we use the multiplicative updating rule
% (Ding et al, 2005)
% For solving H, we use the projected gradient descent method (Hoyer, 2002)
% for that we demand H to have unit norm.

% No global minimum is guaranteed since it has non-convex constraints, and thus
% multiple runs are suggested to achieve the best performance.


% Input:
% 1. V is a m*1 cell including a batch of non-negative symmetric
% matrices of size n*n for factorization, n:# of nodes; m:# of subjects
% 2. rdim is the desired number of clusters/communities, i.e. the reduced
% rank
% 3. b: parameter to control the sparseness. Default:0.1
% 4. maxiter: the maximum times of iterations. Default:5000
% 5. tolvalue: iteration ends if stepsizeH<tolvalue. Default: 1e-14
% 6. fname is the file name to be saved. Default:'test'
% 7. showflag: 1 or 0. 1: show figures of convergence. Default: 0.

% Output:
% H is a n*rdim matrix, representing the group-level membership matrix
% across all subjects, where each column represents a community/cluster/component.
% S is a rdim*m matrix, where each S(j,i) represents the strength of
% component j for subject i, preserving individual differences.
% objhistory recordes the objective function value of each iteration



%by Xuan Li, 2016/11/08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check input arguments
if nargin<2,
    error('Please input Association matrices and the desired rank.');
end
if nargin<3,
    b=0.1;
end
if nargin<4,
    maxiter=5000;
end
if nargin<5,
    tolvalue=1e-14;
end
if nargin<6,
    fname='test';
end
if nargin<7,
    showflag=0;
end

%% Check data
Nsub=length(V);

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
S=abs(randn(rdim,Nsub));

% initialize H
H = abs(randn(vdim,rdim));

% normalize each column of H
H=bsxfun(@rdivide,H,max(abs(H)));

% calculate initial obj value
objhistory=cal_objval(V,S,H,b);




%% Initialize displays
if showflag,
    figure(1); clf; % this will show the energies and sparsenesses
    figure(2); clf; % this will show the objective function value
    drawnow;
end



%% Start iteration

timestarted = clock;
stepsizeH=1e-4; %for gradient descent
iter = 0;

while iter<maxiter,   
    
    % Show progress
    % fprintf('[%d]: %.5f \n',iter,objhistory(end));
    
    % Save every once in a while
    if rem(iter,5)==0,
        elapsed = etime(clock,timestarted);
        %fprintf('Saving...');
        save(fname,'H','iter','objhistory','elapsed');
        %fprintf('Done!\n');
    end
    
    
    %% Plot the progress
    if showflag & (rem(iter,5)==0),
        % Show stats for each column of H    
        figure(1);
        
        % sparsity(h)=( n^0.5-(||h||_1/ ||h||_2))/ (n^0.5-1)
        cursH = (sqrt(vdim)-(sum(H)./sqrt(sum(H.^2))))/(sqrt(vdim)-1);  
        subplot(2,1,2); bar(cursH);
        title('Sparsity')
        xticklabels;
        
        % energy(h)=||h||_2
        title('Energy')
        subplot(2,1,1); bar(sqrt(sum(H.^2)));                           
        
        % Show obj value vs. # of iter
        if iter>1,
            figure(2);
            plot(objhistory(2:end)); 
            title('Objective function value')
            xticklabels;
        end
        drawnow;
    end
    
    % Update iteration count
    iter = iter+1;
    
    % Save old values
    
    Hold{iter} = H;
    Sold{iter} = S;
    
    
    
    
    %% Update H with projected gradient descent
    bat=zeros(vdim,rdim);
    for i=1:Nsub
        S_i=diag(S(:,i));
        bat = bat+(H*S_i*H'*H*S_i-V{i}*H*S_i);
    end
    dH = bat+b;
    
    begobj = objhistory(end);
    
    % Make sure we decrease the objective!
    while 1,
        %step 1: gradient desecnt
        Hnew = H - stepsizeH*dH;
        
        %step 2: set negative values to zero
        Hnew(find(Hnew<0))=0;
        
        %step 3: normalized H
        %Hnew = Hnew./(ones(vdim,1)*sqrt(sum(Hnew.^2))); 
        Hnew=bsxfun(@rdivide,Hnew,max(abs(Hnew)));
        
        %step 4: recal obj value
        newobj=cal_objval(V,S,Hnew,b);

        
        % If the objective decreased, we can continue to update S
     
        if newobj<=begobj,
            break;
        end
        %disp(['old obj= ',num2str(begobj),'!']);
        %disp(['new obj= ',num2str(newobj),'!']);
        
        % else decrease stepsize and try again
        stepsizeH = stepsizeH/2;
        %fprintf('.');
        
        if stepsizeH<tolvalue,
            %fprintf('Algorithm converged.\n');
            return;
        end
    end
    
    % Slightly increase the stepsize
    stepsizeH = stepsizeH*1.2;
    H = Hnew;
       
    
    %% Update S with multiplicative step
    
    for i=1:Nsub
        S_i=diag(S(:,i));
        S_new(:,i)=diag(S_i.*(H'*V{i}*H)./(H'*H*S_i*H'*H+1e-9)); 
    end
    % le-9 to avoid zero in the denominator
    
    S=S_new;
    
    % Calculate obj value
    newobj=cal_objval(V,S,H,b);
    
    %disp(['update S'])  ;
    objhistory = [objhistory newobj];
end


%%%%
function objval=cal_objval(V,S,H,b)
sum_err=0;
Nsub=length(V);
for i=1:Nsub
    S_i=diag(S(:,i));
    sum_err=sum_err+sum(sum((V{i}-H*S_i*H').^2));
end
objval = 0.5*sum_err+b*sum(sum(H));
        