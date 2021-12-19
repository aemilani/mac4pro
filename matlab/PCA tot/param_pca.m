true_signal=load('validation.dat'); 
[N,n]=size(true_signal);
v=[0.95,0.98,0.99,0.995,1]

set(0,'DefaultFigureVisible','off');% all subsequent figures "off”

for i=1:5
    i
    [test_reconstruction_n,n_PC(i),rmse(i)]=PCA_reconstruction('train.dat','validation.dat',v(i));
    [test_reconstruction,n_PC(i),no_int]=PCA_reconstruction('train.dat','validation_sim.dat',v(i));
    robustness(i)=sqrt(sum((test_reconstruction(:,6)-true_signal(:,6)).^2))/(N);
    acc(i)=sqrt(sum((test_reconstruction_n(:,6)-true_signal(:,6)).^2))/(N);
    close all
end


set(0,'DefaultFigureVisible','on'); % all subsequent figures "on"
figure
% subplot(211)
plot(n_PC,acc(1:5),'.')
hold on
plot(n_PC,robustness(1:5),'r.')

legend('reconstruction error','robustness')
