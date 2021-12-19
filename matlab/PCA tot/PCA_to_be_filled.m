function [test_reconstruction,n_PC]=PCA_reconstruction2(train_data_name,test_data_name,perc_var)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  INPUT:
%       train_data_name = name of the file that contains the training patterns 
%       test_data_name = name of the file that contains the test patterns
%       perc_var  = parameter used to find how many Principal Components
%                   are  kept in the PCA approximation. In practice, the
%                   eigenvalue are sorted from the largest to the smallest
%                   and we add Principal Components untill we reach perc_var
%           
%OUTPUT:
%       test_reconstruction = reconstruction of the test patterns (i.e. their 
%                             expected values in normal conditions)
%       n_PC = number of principal components kept in the PCA approximation
%       figures reporting for each signal the measured value, the
%       reconstruction and the residuals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load train and test data
train_data=load(train_data_name);
test_data=load(test_data_name);
%--------------------------------------------------------------------------
[N,n]=size(train_data); %N is the number of training pattern, n is the number of signals
var_th=perc_var; %in the pca approximation, Minimum fraction variance component to be kept
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% DATA PRETREATMENT
%--------------------------------------------------------------------------
%normalize training_data: mean of each signal is 0, standard deviation is 1
m_data=mean(train_data);
std_data=std(train_data);
for isig=1:n
    train_data_n(:,isig)=(train_data(:,isig)-m_data(isig))/std_data(isig);
end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% TRAINING PHASE TO BE FILLED
%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    STEP 1: find the matrix P whose column are the eigenvectors ordered
%    from the largest (column 1) to the smallest (column n)
%    SIZE of P is [n,n]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    STEP 2: perform the PCA approximation (keep a number (lambda) of 
%            eigenvestors such that they represent at least perc_var of the
%            total data variance)
%               =
%            find the matrix P_lambda whose columns are the first lambda aigenvectors 
%            Assuming to keep only lambda PCs, size of P_lambda is [n,lambda] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%end training phase
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% TEST PHASE (here perform the reconstruction of the test patterns)
%--------------------------------------------------------------------------
[N_test,n]=size(test_data);

%normalize test data according to mean and standard deviation computed on
%the training data


for isig=1:n
    test_data_n(:,isig)=(test_data(:,isig)-m_data(isig))/std_data(isig);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      TO BE FILLED
%
%     RECONSTRUCT THE TEST PATTERNS
%     Call 'test_data_rec_n' the matrix containing the reconstruction of 
%     the test patterns. Size of test_data_rec_n is [N_test,n]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%denormalize
for isig=1:n
    test_data_rec(:,isig)=test_data_rec_n(:,isig)*std_data(isig)+m_data(isig);
end

test_reconstruction=test_data_rec;

%plot figures
signal_name=['T1';'T2';'T3';'T4';'T5';'T6'];

measure_error=0.3;
[N_test,n]=size(test_data);
for i=1:6;
    figure;
    subplot(2,1,1)
    plot(test_data(:,i),'b');
    hold on
    plot(test_data_rec(:,i),'r');
    xlabel(signal_name(i,:));
    legend('true value','reconstruction')
    subplot(2,1,2);
    plot(test_data_rec(:,i)-test_data(:,i),'r');
    legend('residual');
    hold on
    upper_error=ones(N_test)*3*measure_error;
    lower_error=-ones(N_test)*3*measure_error; 
    plot(upper_error);
    hold on
    plot(lower_error);
    ylabel('residual');
    xlabel(signal_name(i,:));
    axis([0 N_test -10*measure_error 10*measure_error])
end
