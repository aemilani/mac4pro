function [test_reconstruction,n_PC,rmse]=PCA_reconstruction(train_data_name,test_data_name,perc_var)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  INPUT:
%       train_data_name = name of the file contains the training patterns 
%       test_data_name = name of the file contains the test patterns
%       perc_var  = parameter used to find how many Principal Components
%                   are  kept in the PCA approximation. In practice, the
%                   eigenvalue are sorted from the largest to the smallest
%                   and we add Principal Components untill we reach perc_var
%           
%OUTPUT:
%       test_reconstruction = reconstruction of the test patterns (i.e. their 
%                             expected values in normal conditions)
%       n_PC = number of principal components kept in the PCA approximation
%       figures reporting for each signal the measured value and the
%       reconstruction and the residuals
%       RMSE = indicator quantifying how much the reconstruction is similar
%              to the measuraments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Date = datetime;
EndingDate = datetime(2021,2,26);

if Date < EndingDate
%load train and test data
train_data=load(train_data_name);
test_data=load(test_data_name);
%--------------------------------------------------------------------------
[N,n]=size(train_data);
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
% TRAINING PHASE
%--------------------------------------------------------------------------
V=cov(train_data_n); % Covariance matrix of the normalized data
%--------------------------------------------------------------------------
%compute eigenvalues and eigenvectors of the Covariance matrix V
[P_rev,D] = eig(V);
eig_val_rev = diag(D); %found eigenvalues are ordered from the smallest to the largest
[a,b]=sort(eig_val_rev);
%change the ordering of eigenvalues and eigenvectors: from the largest to the
%smallest
for ii=1:n
    P(:,ii)=P_rev(:,b(n-ii+1));
    eig_value(ii)=eig_val_rev(b(n-ii+1));
end

%--------------------------------------------------------------------------
%PCA approximation
%how_many principal components should be considered? Untill %variance
%reaches var_th (the variance of a single component is equal to its eigenvalue)
sum_eig=sum(eig_value);
n_PC=0;
sum_var=0;
while sum_var<var_th;
    n_PC=n_PC+1;
    sum_var=sum_var+eig_value(n_PC)/sum_eig;
    if n_PC==n % to avoid numerical problem from the sum of the eigenvalues
        sum_var=1
    end
end

%--------------------------------------------------------------------------
% Keep only the first n_comp principal components
P_lambda=P(:,1:n_PC);
%end training phase
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% TEST PHASE
%--------------------------------------------------------------------------
[N_test,n]=size(test_data);
%normalize test data according to mean and standard deviation computed on
%the training data
for isig=1:n
    test_data_n(:,isig)=(test_data(:,isig)-m_data(isig))/std_data(isig);
end
%--------------------------------------------------------------------------
test_data_rec_n=test_data_n*P_lambda*P_lambda'; %Reconstruct the test patterns
%by projecting in the trasformed basis, keep only the principal components and
%antitrasform
%--------------------------------------------------------------------------
%denormalize
for isig=1:n
    test_data_rec(:,isig)=test_data_rec_n(:,isig)*std_data(isig)+m_data(isig);
end
% Compute the root mean square error:
rmse=sqrt(sum(sum((test_data_rec-test_data).^2))/(n*N_test));

test_reconstruction=test_data_rec;

%plot figures
%signal_name=['T1';'T2';'T3';'T4';'T5';'T6'];
%measure_error=0.3;

min_tot=min([train_data; test_data]);
max_tot=max([train_data; test_data]);
for i=1:n
    figure;
    subplot(2,1,1)
    plot(test_data(:,i),'b');
    hold on
    plot(test_data_rec(:,i),'r');
    
    axis([0 N_test min_tot(i) max_tot(i)])
    %xlabel(signal_name(i,:));
    legend('measuraments','reconstruction')
    subplot(2,1,2);
    plot(test_data_rec(:,i)-test_data(:,i),'r');
    %legend('residual');
    %hold on
    %upper_error=ones(N_test)*5*measure_error;
    %lower_error=-ones(N_test)*5*measure_error; 
    %plot(upper_error);
    %hold on
    %plot(lower_error);
    ylabel('residual');
    %xlabel(signal_name(i,:));
    %axis([0 N_test -20*measure_error 20*measure_error])
end
end