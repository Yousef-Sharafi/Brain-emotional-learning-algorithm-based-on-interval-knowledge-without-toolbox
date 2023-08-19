
clc;
clear all;
close all;

data=xlsread('LORRENZ.xlsx');
data = (data - min(data))/(max(data)-min(data));
data1=data;
depth = 2;
n = numel(data)-depth;
data_input=zeros(n,depth);
target=zeros(n,1);
output=zeros(n,1);
for i =1:size(data,1)-depth
    for j = 1: depth
        data_input(i,j)=data(i+j-1);
    end
    target(i,1)=data(i+depth);
end
data=data_input;
epoch=100;
number_train=round(0.75*n);
number_test=n-number_train;
eta=0.1;
eta_m=0.1;
rew=2;
vi=unifrnd(-1,1,[1,2]);
wi=unifrnd(-1,1,[1,2]);
Ai=unifrnd(-1,1,[1,2]);
Oi=unifrnd(-1,1,[1,2]);

for iter=1:epoch
    
   for i=1:number_train     
       x=data(i,1);
       y=data(i,2);
       z=target(i);
       max_s=max(x,y);
       Ai(1)=x*vi(1);
       Ai(2)=y*vi(2);
       Oi(1)=x*wi(1);
       Oi(2)=y*wi(2);      
       E=(sum(Ai)+max_s)-sum(Oi);
       error=z-E;
       vi(1)= vi(1)+error*eta*(x*max(0,rew-sum(Ai)));
       vi(2)= vi(2)+error*eta*(y*max(0,rew-sum(Ai))); 
       wi(1)= wi(1)+error*eta_m*(x*(Oi(1)+Oi(2)-2*rew));
       wi(2)= wi(2)+error*eta_m*(y*(Oi(1)+Oi(2)-2*rew));
      
   end
   
 for i=1:number_train     
       x=data(i,1);
       y=data(i,2);
       z=target(i);
       max_s=max(x,y);
       Ai(1)=x*vi(1);
       Ai(2)=y*vi(2);
       Oi(1)=x*wi(1);
       Oi(2)=y*wi(2);      
       E=(sum(Ai)+max_s)-sum(Oi);
       output(i)=E;
 end


figure(1);
subplot(1,2,1),plot(output(1:number_train),'-b');
hold on;
subplot(1,2,1),plot(target(1:number_train),'-r');
hold off;
mse1=mse(output(1:number_train)-target(1:number_train))
title(sprintf('Brain Emotional Learning + Train \nEpoch = %d    MSE = %.10f ',iter,mse1),'fontsize',10,'fontweight','b');

 for i=1:number_test     
       x=data(number_train+i,1);
       y=data(number_train+i,2);
       z=target(number_train+i);
       max_s=max(x,y);
       Ai(1)=x*vi(1);
       Ai(2)=y*vi(2);
       Oi(1)=x*wi(1);
       Oi(2)=y*wi(2);      
       E=(sum(Ai)+max_s)-sum(Oi);
       output(number_train+i)=E;
 end

subplot(1,2,2),plot(output(number_train+1:n),'-b');
hold on;
subplot(1,2,2),plot(target(number_train+1:n),'-r');
hold off;

figure(2);
mse1=mse(output(number_train+1:n)-target(number_train+1:n));
title(sprintf('Brain Emotional Learning - Test\n    MSE = %.10f ',mse1),'fontsize',10,'fontweight','b');
end
x1=output(number_train+1:n);
y1=target(number_train+1:n);
hand = plotregression(x1, y1, 'Regression');
h = get(hand, 'Children');
hh = get(h(2), 'Children');
delete(hh(3))
set(hh(1), 'Marker', '*')
legend('Data', 'Fit', 'Location', 'NorthWest');

