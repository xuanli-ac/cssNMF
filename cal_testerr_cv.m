function [te, Results_train, Results_test]=cal_testerr_cv(Mtest,Mtrain,k,b,nrun,maxiter1,maxiter2,tolv1,tolv2,showflag)

% This function computes the test error for parameter selection of cssNMF
% with cross-validation.

% Input:
% Mtest: testing set, association matrices of test subjects
% Mtrain: training set, association matrices of training subjects
% k: number of communities/components
% b: sparsity degree. Default 0.1
% nrun: number of runs to select the best solution for cssNMF. Default 20
% maxiter1/tolv1 & maxiter2/tolv2: parameters required by css_nmf.m and
% css_nmf_S.m respectively, maximum # of iterations and stop criteria


% Output?
% te: test error=reconstruction error on testing set/variance
% Results_train: H, S, obj, obj_end record the results over all runs; H_br,
% S_br are the best among these runs with minimum obj value.
% Results_test: St, objt, objt_end, derived on the testing set.

if nargin<2,
    error('Please input the training set, testing set, desired rank and sparsity');
end
if nargin<4,
    b=0.1;
end
if nargin<5,
    nrun=20;
end
if nargin<6,
    maxiter1=5000;
end
if nargin<7,
    maxiter2=100;
end
if nargin<8,
    tolv1=1e-14;
end
if nargin<9,
    tolv2=1e-16;
end


%% calculate the variance of association matrices
mvar=0;
[Nnode,~]=size(Mtest{1});
Mmean=zeros(Nnode,Nnode);
for i=1:length(Mtest)
    Mmean=Mmean+Mtest{i};
end

Mmean=Mmean./length(Mtest);

for i=1:length(Mtest)
    mvar=mvar+sum(sum((Mtest{i}-Mmean).^2));
end



            
%% Training part

S=cell(1,nrun);
obj=cell(1,nrun);
H=cell(1,nrun);

for r=1:nrun
    [H{r},S{r},obj{r}]= css_nmf( Mtrain, k, b,maxiter1,tolv1, 'train',showflag );
    obj_end(r) = obj{r}(end);
end

% select the best run with min obj val
[~,q]=min(obj_end); %q is the best run with min obj
H_br=H{q};
S_br=S{q};

% save results
Results_train.S=S;
Results_train.obj=obj;
Results_train.H=H;
Results_train.H_br=H_br;
Results_train.S_br=S_br;
Results_train.obj_end=obj_end;


%% Testing part

%obtain corresponding S
[Stest,objtest]= css_nmf_S(Mtest, H_br,maxiter2,tolv2,showflag);
objt_end=objtest(end);

% calculate the test error: reconstruction error/variance
te=objt_end./mvar;

% save results
Results_test.St=Stest;
Results_test.objt=objtest;
Results_test.objt_end=objt_end;


