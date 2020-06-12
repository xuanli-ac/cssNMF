# collective sparse symmetric non-negative matrix factorization（cssNMF）
>Matlab code for overlapping community detection in brain functional network

The cssNMF method is a NMF-based technique, designed to detect the overlapping community structure in brain functional networks. Given a batch of non-negative and symmetric association/similarity matrices, cssNMF collectively tri-factorizes these matrices in a symmetric way, into a membership matrix H and a weight matrix S. Each column of H represent a group-level component/community, where each element H_ij indicates the importance of feature i in component j. Each element S_ij represents the strength of component i in the representation of the j-th association matrix. 

When applied to brain functional networks derived on fMRI data, cssNMF detects the group-level overlapping communities H shared by all subjects and reserves individual differences in the weight matrix S. Specifically, given the non-negative and symmetric association matrices (each of size N by N, where N is the number of nodes) of m subjects, this code returns the group-level overlapping community structure H (N by k) and a weight matrix S (k by m), where S_ij represents the strength of the i th community in the brain functional network of subject j.


## Description
* css_nmf.m - perform the cssNMF algortihm to factorize a group of association matrices into a membership matrix and a weight matrix
* css_nmf_S.m - learn the corresponding weight matrix S given the membership matrix H by cssNMF. It is used in cross-validation.
* cal_testerr_cv.m - calculate the test error of cross validation. It is used for parameter selection.
* main_parameter_selection.m - perform the parameter selection by using grid search with nfold cross-validation. The membership matrix H is learned on training set and used to obtain the corresponding weight matrix S on testing set.
* Associaion_Matrices - folder consists of the association matrices of a group of subjects for testing the algorithm, M.

## Usage
* use css_nmf.m to detect the group-level overlapping community structure and individual differences in community strength
```
[H,S,objhistory]= css_nmf( V, rdim, b,maxiter,tolvalue, fname, showflag )

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
```

* run main_parameter_selection.m 
It calculates the test error of cross validation with different combination of parameters (saved as TE.mat in a folder named 'Results_cv') and plots the results for parameter selection.
```
% This main function is used to select appropriate parameters (k: desired
% rank and b: sparsity degree) for cssNMF

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
```

## Reference
- Li, X., Gan, J. Q., and Wang, H. (2018). **[Collective sparse symmetric non-negative matrix factorization for identifying overlapping communities in resting-state brain functional networks.](https://www.sciencedirect.com/science/article/abs/pii/S1053811917309102)** *NeuroImage*, vol. 166, pp. 259–275.

- Hoyer, P.O. (2002). **[Non-negative sparse coding.](https://ieeexplore.ieee.org/document/1030067)** *In: Proceedings of the 12th IEEE Workshop on Neural Networks for Signal Processing*, pp. 557–565.

- Hoyer, P.O. (2004). **[Non-negative matrix factorization with sparseness constraints.](http://www.jmlr.org/papers/v5/hoyer04a.html)** *J. Mach. Learn. Res.*, vol. 5, pp. 1457–1469.

- Ding, C.H., He, X., Simon, H.D. (2005). **[On the equivalence of nonnegative matrix
factorization and spectral clustering.](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972757.70)** *In: Proceedings of the 5th SIAM International Conference on Data Mining*, pp. 606–610.
