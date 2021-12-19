clear all 
close all
true_signal=load('validation.dat'); 
[N,n]=size(true_signal);
v=[0.98,0.99,0.995,1]
for i=1:4;
    [test_reconstruction,n_PC(i),rmse(i)]=PCA_reconstruction('train.dat','validation.dat',v(i));
close all
end
 
figure(1)
% subplot(211)
plot(v,n_PC,'.')
legend('number of principal components')
figure(2)
plot(v,rmse,'r.')
legend('RMSE')

