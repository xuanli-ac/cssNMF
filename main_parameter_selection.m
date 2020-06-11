% This main function is used to select appropriate parameters (k: desired
% rank and b: sparsity degree) for cssNMF


clc;
clear;
load('Association_Matrices/M.mat')

% klist=2:10;
% blist=0.1:0.1:1;


%% initializing parameters (need be adjusted manually)

klist=5:1:10; 
blist=0.1:0.1:0.5; 
nfold=2;
nrun=10;
maxiter1=1000;
maxiter2=50;
tolv1=1e-14;
tolv2=1e-16;
showflag=0;

fname=['Results_cv',num2str(nfold),'folds'];

%% partition the dataset into n folds for cross validation
[Mtrain,Mtest,ind_train,ind_test] = separate_train_test_data(nfold,M);



%% Start grid search for b (sparsity parameter) and k (reduced rank)
nb=1;
for b=blist
    
    % save results in folder 'Results_cv' and create a subfolder for each b
    fname1=[fname,'/b_',num2str(b)];
    if ~exist(fname1,'dir')
        mkdir(fname1);
    end
    
    nk=1;
    for k=klist
        
        te=0;
        Results_train=[];
        Results_test=[];
        
        for f=1:nfold
            
            [te(f), Results_train{f}, Results_test{f}]=cal_testerr_cv(Mtest{f},Mtrain{f},k,b,nrun,maxiter1,maxiter2,tolv1,tolv2,showflag);
            save([fname1,'/Results_train_k',num2str(k),'.mat'],'Results_train')
            save([fname1,'/Results_test_k',num2str(k),'.mat'],'Results_test')
        end
        
        TE(nb,nk)=mean(te);
        nk=nk+1;
        save([fname,'/TE.mat'],'TE')
        disp(['finish b=',num2str(b),' k=',num2str(k),'!']);
    end
    nb=nb+1;
    
end

%% plot TE

figure;
subplot(1,2,1)
for i=1:length(blist)
    lglist{i}=['\beta=',num2str(blist(i))];
end
plot(klist,TE,'o-','LineWidth',1.5,'MarkerSize',3);
title('Test error vs k','FontWeight','normal');
xlabel('Number of communities k')
ylabel('Test error')
legend(lglist)

subplot(1,2,2)
for i=1:length(klist)
    lglist2{i}=['k=',num2str(klist(i))];
end
plot(blist,TE,'o-','LineWidth',1.5,'MarkerSize',3);
title('Test error vs \beta','FontWeight','normal');
xlabel('sparsity level \beta')
ylabel('Test error')
legend(lglist2)


%%%% partition dataset into training set and testing set
function [Vtrain,Vtest,ind_train,ind_test] = separate_train_test_data(nfold,V)
Vtrain=cell(1,nfold);
Vtest=cell(1,nfold);
[~,Nsub]=size(V);
L=1:Nsub;
t=floor(Nsub/nfold);
for i =1:nfold
    
    if i==1
        Vtrain{i}=V((t+1):end);
        ind_train{i}=L((t+1):end);
    else
        if i==nfold
            Vtrain{i}=V(1:(end-t));
            ind_train{i}=L(1:(end-t));
        else
            Vtrain{i}=[V(1:(i-1)*t) V(i*t+1:end)];
            ind_train{i}=[L(1:(i-1)*t); L(i*t+1:end)];
        end
    end
    Vtest{i}=V((i-1)*t+1:i*t);
    ind_test{i}=L((i-1)*t+1:i*t);
end
end