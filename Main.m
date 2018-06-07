close all; clear all; clc;

GT = csvread('series_filtfilt.csv');
Pred = csvread('FFNN_predictions10ahead.csv');

nz = find(Pred~=0);
Pred_new = zeros(length(nz)+60,1);
Pred_new(60:length(Pred_new)-1,1)=Pred(nz,1);
GT = GT(1:length(Pred_new));

Pred = Pred_new;

bursts_GT=[];
bursts_pred=[];
pred_thresh = [0,0.5,1.0,1.5,1.96];
pred_GT = 1.96;
for x=1:length(GT)/50
   bursts_GT(x)=DetectBurst(GT(50*(x-1)+1:50*x),pred_GT,3,10); 
   bursts_pred(1:5,x)=DetectBurst(Pred(50*(x-1)+1:50*x),pred_thresh,3,10);
end
    
figure();
a = [1:50:length(GT)-50];
%plot(Pred,'r');hold on;
plot(GT,'b');hold on;
plot(a,2*bursts_GT,'b*')
%hold on;
%plot(a,1.8*bursts_pred,'r*');
xlim([14000 15000]);



%plot(Pred,'r')
%hold on;
%plot(GT,'b')
%hold on;
% a - true positive (GT 1 and Pred 1)
% b - false negative (GT 1 and Pred 0)
% c - false positive (GT 0 and Pred 1)
% d - true negative (GT 0 and Pred 0)
for j=1:5
    c4=1;c3=1;c2=1;c1=1;
    for i=1:length(bursts_GT)
        if bursts_GT(1,i)==0 && bursts_pred(j,i)==0
            d(c4)=1;
            c4 = c4+1;
        end
        if bursts_GT(1,i)==0 && bursts_pred(j,i)==1
            c(c3)=1;
            c3=c3+1;
        end
        if bursts_GT(1,i)==1 && bursts_pred(j,i)==0
            b(c2)=1;
            c2=c2+1;
        end
        if bursts_GT(1,i)==1 && bursts_pred(j,i)==1
            a(c1)=1;
            c1=c1+1;
        end
    end
    sens = length(a)/(length(a)+length(b));
    spec = length(d)/(length(c)+length(d));
    clear a b c d;
    tp_rate = sens; %yaxis
    fp_rate = 1-spec; %xaxis

    rocx(j,1)=fp_rate;
    rocy(j,2)=tp_rate;
end

figure()
plot(rocx,rocy);
hold on;
plot([0:0.2:1],[0:0.2:1],'r-')

