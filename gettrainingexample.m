clear all; close all; clc;


prelen = 50;            % Number of passed time samples (50 ms) for input
forward = 10;           % Number of time samples forward to predict (10 ms)
filt_xcld = 50;         % Duration at the edge of segments to be excluded (50 ms)
DetectWindow = 50;      % Detection window (50 ms width)
chunksize = prelen+forward+DetectWindow-1;
truth_thresh = 0.349;	% Ground truth detection threshold
Nconsc = 3;             % Number of consecutive peaks
mindist = 10;           % Minimum distance between two peaks
close all
rng(111);

fs = 1000;

%delta(<5) theta(5,8) alpha(9,14) beta(15,40) 
%gamma(64,84) high gamma(84,124)
oscBand = [1,10;5,8;9,14;15,40;64,84;84,124];
n_bands = length(oscBand)+1; %Add one for raw
bFilt = zeros(n_bands,5);
aFilt = zeros(n_bands,5);


%lowpass filter
[b a] = butter(2,oscBand(1,2)/(fs/2),'low');
bFiltlow=b;
aFiltlow=a;
clear b a

%bandpass filters
for i=2:6
    [b a] = butter(2,oscBand(i,:)/(fs/2));
    bFilt(i,:)=b;
    aFilt(i,:)=a;
    clear b a
end

% load the data
load('subject4.mat');
%LFP = subject5_noart; %This line is needed if loading from csv
T = length(LFP);

% Get start time points of each chunk
xcld = 200;             % Exclude 0 periods more than 200 ms
minLen = 1000;          % Minimum length for a segment
seg = getseg(LFP,xcld,minLen);  % get segment time
nseg = size(seg,2);     % number of segments
ts = cell(1,nseg);      % start time point of chunks
valid = false(1,T);     % mark for valid segments
for i = 1:nseg
    ts{i} = seg(1,i)+1+filt_xcld:seg(2,i)-filt_xcld-chunksize+1;
    valid(seg(1,i)+1:seg(2,i)) = true;
end
ts = cell2mat(ts);
Ttot = length(ts);

% Select training samples
prop = 0.1;                 % Proportion of training samples (10%)
ntrain = round(Ttot*prop);  % number of training samples
t = sort(randsample(ts,ntrain));

% filtfilt for ground truth. causal filter for training input.
ZS = (LFP-mean(LFP(valid)))/std(LFP(valid));    % z-score

ZS_causal = zeros(n_bands,length(ZS));
ZS_noncausal = zeros(n_bands,length(ZS));

%Filter low pass
ZS_causal(1,:) = filter(bFiltlow,aFiltlow,ZS);
ZS_noncausal(1,:) = filtfilt(bFiltlow,aFiltlow,ZS);

%Filter band pass
for i=2:n_bands-1
    ZS_causal(i,:) = filter(bFilt(i,:),aFilt(i,:),ZS);
    ZS_noncausal(i,:) = filtfilt(bFilt(i,:),aFilt(i,:),ZS);
end
ZS_causal(n_bands,:) = ZS; % The last "channel" is the raw
ZS_noncausal(n_bands,:) = ZS; % The last "channel" is the raw


colors = ['b','r','g','m','m','c','k'];
    
%ZS_gamma = ZS_causal(find(oscBand(:,1)==64),:); %gamma is the one with 64Hz
ZS_gamma = ZS_noncausal(find(oscBand(:,1)==64),:);

% x and x_long are the matrices for NN training and spike triggered average
x = zeros(ntrain,prelen+2,n_bands); % 3D matrix ntrain x prelen+2 x n_bands
x_longcausal = zeros(ntrain,chunksize,n_bands); % 3D matrix with entire region of interest
x_longnc = zeros(ntrain,chunksize,n_bands); % 3D matrix with entire region of interest

progress = 0;
for i = 1:ntrain
    
    % detectwindow is where the burst gets detected.
    dw = ZS_gamma(t(i)+prelen+forward-1:t(i)+chunksize-1);
    [alarm,t1pk] = DetectBurst(dw,truth_thresh,Nconsc,mindist);
    
    if ~alarm
        for j = 1:n_bands
            x_longcausal(i,:,j) = ZS_causal(j,t(i):t(i)+chunksize-1);
            x_longnc(i,:,j) = ZS_noncausal(j,t(i):t(i)+chunksize-1);
            
            x(i,1:prelen,j) = ZS_causal(j,t(i):t(i)+prelen-1);
            x(i,prelen+1,j) = alarm;
            x(i,prelen+2,j) = t1pk;
        end
    end
    
    %If alarm is on, adjust training window to line up to first peak
    if alarm
        for j = 1:n_bands
            x_longcausal(i,:,j) = ZS_causal(j,t(i)+(t1pk-1):t(i)+chunksize+(t1pk-1)-1);
            x_longnc(i,:,j) = ZS_noncausal(j,t(i)+(t1pk-1):t(i)+chunksize+(t1pk-1)-1);
            
            x(i,1:prelen,j) = ZS_causal(j,t(i)+(t1pk-1):t(i)+prelen+(t1pk-1)-1);
            x(i,prelen+1,j) = alarm;
            x(i,prelen+2,j) = t1pk;
        end
    end
    
    if progress~=round(i/ntrain*100)
        progress = round(i/ntrain*100);
        disp([num2str(progress),'% completed.']);
    end
end
disp([num2str(sum(x(:,prelen+1,4))),' out of ',num2str(ntrain),' samples have burst.']);
% csvwrite('train_sub4.csv',x);	% create training set file

figure

x_on = find(x(:,51,1)==1);
x_off = find(x(:,51,1)==0);

%Plot only gamma band

%Plot a few examples in grey
r = randi(100,1,10);
subplot(2,1,1)
for i=1:length(r)
    plot(x_longnc(x_on(r(i)),:,find(oscBand(:,1)==64)),'Color',[0.4 0.4 0.4],'LineWidth',0.2);
    hold on;
end

x_avg = mean(x_longnc(x_on,:,find(oscBand(:,1)==64)));
plot(x_avg,'g','LineWidth',3);
title('non-causal filter')
ylim([-1 1])

subplot(2,1,2)
for i=1:length(r)
    plot(x_longcausal(x_on(r(i)),:,find(oscBand(:,1)==64)),'Color',[0.4 0.4 0.4],'LineWidth',0.2);
    hold on;
end

x_avg = mean(x_longcausal(x_on,:,find(oscBand(:,1)==64)));
plot(x_avg,'m','LineWidth',3);
ylim([-1 1])
title('causal filter')
xlabel('time (ms)')
% plot(50*ones(1,15),[-0.7:0.1:0.7],'r-') %vertical lines to show boundaries o
% plot(60*ones(1,15),[-0.7:0.1:0.7],'r-')
% plot(110*ones(1,15),[-0.7:0.1:0.7],'r-')
% ylim([-0.7 0.7])



%Plot rest of the bands
figure
subplot(2,1,1)
for i=1:n_bands
    gam = find(oscBand(:,1)==64);
    if i~=gam
        x_avg = mean(x_longcausal(x_on,:,i));
        plot(x_avg,colors(i))
        hold on;
    end
end
legend('lowpass (<10Hz)','theta (5 to 8Hz)','alpha (9 to 14Hz)','beta (15 to 40 Hz)','high gamma (84 to 124Hz)','raw');
ylim([-1 1])
title('gamma burst')
% plot(50*ones(1,15),[-0.7:0.1:0.7],'r-') %vertical lines to show boundaries o
% plot(60*ones(1,15),[-0.7:0.1:0.7],'r-')
% plot(110*ones(1,15),[-0.7:0.1:0.7],'r-')
% ylim([-0.7 0.7])
hold off;

%Plot all bands for no burst
subplot(2,1,2)
for i=1:n_bands
    x_avg = mean(x_longcausal(x_off,:,i));
    plot(x_avg,colors(i))
    hold on;
end
ylim([-1 1])
title('no gamma burst')
xlabel('time (ms)')


figure
%%% What's up with the raw?
%Plot a few examples in grey
r = randi(100,1,100);
for i=1:length(r)
    plot(x_longcausal(x_on(r(i)),:,7),'Color',[0.4 0.4 0.4],'LineWidth',0.2);
    hold on;
end
%Plot the average in magenta
x_avg = mean(x_longnc(x_on,:,7));
plot(x_avg,'m','LineWidth',3);
hold on;
% plot(50*ones(1,15),[-0.7:0.1:0.7],'r-') %vertical lines to show boundaries o
% plot(60*ones(1,15),[-0.7:0.1:0.7],'r-')
% plot(110*ones(1,15),[-0.7:0.1:0.7],'r-')
%ylim([-0.7 0.7])
% figure
% hold on
% plot(ZS_causal(4,:),'b');
% iburst = find(x(:,prelen+1,4));
% tpk = t(iburst)-1+forward+prelen+x(iburst,end,4)';
% plot(tpk,ZS_causal(4,tpk),'r.');
% plot([1,T],[1,1]*truth_thresh,'g');
% axis tight;
% legend('filt signal','1st burst peaks','detection threshold');

%csvwrite('binarygammafilt.csv',x(:,:,5))
%csvwrite('binarythetafilt.csv',x(:,:,2))
%csvwrite('binaryhighgammafilt.csv',x(:,:,6))

% Plot spike triggered average
%figure
%plot(mean(x(find(x(:,51)==1),1:50)),'b');hold on;%gamma
%plot(mean(x2(find(x2(:,51)==1),1:50)),'r');hold on;%raw
%plot(mean(x3(find(x3(:,51)==1),1:50)),'g');hold on;%low

