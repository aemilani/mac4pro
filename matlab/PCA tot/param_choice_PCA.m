true_signal=load('validation.dat'); 
[N,n]=size(true_signal);
v=[0.95,0.98,0.99,0.995,1];
for i=1:4
    [test_reconstruction,n_PC(i),rmse(i)]=PCA_reconstruction('train.dat','validation.dat',v(i));
    [test_reconstruction,n_PC(i),no_int]=PCA_reconstruction('train.dat','val_anomaly.dat',v(i));
    robustness(i)=sqrt(sum(sum((test_reconstruction(:,1)-true_signal(:,1)).^2))/(N));
    close all
end

figure
% subplot(211)
plot(n_PC,rmse,'.')
hold on
plot(n_PC,robustness,'r.')

legend('reconstruction error','robustness')
