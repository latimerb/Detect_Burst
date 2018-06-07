clear all;  close all;
prelen = 50;            % Number of passed time samples (50 ms) for input
forward = 10;           % Number of time samples forward to predict (10 ms)
filt_xcld = 50;         % Duration at the edge of segments to be excluded (50 ms)
DetectWindow = 50;      % Detection window (50 ms width)
chunksize = prelen+forward+DetectWindow-1;
truth_thresh = 0.349;	% Ground truth detection threshold
Nconsc = 3;             % Number of consecutive peaks
mindist = 10;           % Minimum distance between two peaks
endratio = 0.25;        % Minimum ratio of the last peak value to the maximum peak value in a burst
rng(111);

fs = 1000;
oscBand = [1,5;5,8;9,14;15,40;64,84;84,124];    % Filter bands
gamma_band = 5;
n_bands = size(oscBand,1);

load(fullfile('dataset','subject4.mat'));
%load('subject4.mat');
T = length(LFP);

%% Get start time points of each chunk
xcld = 200;             % Exclude 0 periods more than 200 ms
minLen = 1000;          % Minimum length for a segment
seg = getseg(LFP,xcld,minLen);  % get segment time
nseg = size(seg,2);     % number of segments
ts = cell(1,nseg);      % start time point of chunks
tseg = zeros(2,nseg);   % valid time domain of a burst
valid = false(1,T);     % mark for valid segments
for i = 1:nseg
    ts{i} = seg(1,i)+1+filt_xcld:seg(2,i)-filt_xcld-chunksize+1;
    tseg(:,i) = [seg(1,i)+1+filt_xcld+prelen+forward;seg(2,i)-filt_xcld];
    valid(seg(1,i)+1:seg(2,i)) = true;
end
ts = cell2mat(ts);
Ttot = length(ts);

% Select training samples
prop = 0.1;                 % Proportion of training samples (10%)
ntrain = round(Ttot*prop);  % number of training samples
t = sort(randsample(ts,ntrain));

%% filtfilt for ground truth. causal filter for training input.
ZS = (LFP-mean(LFP(valid)))/std(LFP(valid));    % z-score
ZS_causal = zeros(T,n_bands+1);
[bFilt,aFilt] = butter(2,oscBand(1,2)/(fs/2),'low');
ZS_causal(:,1) = filter(bFilt,aFilt,ZS);
for i = 2:n_bands
    [bFilt, aFilt] = butter(2,oscBand(i,:)/(fs/2));
    ZS_causal(:,i) = filter(bFilt,aFilt,ZS);
    if i == gamma_band
        ZS_gamma = filtfilt(bFilt,aFilt,ZS);
    end
end
ZS_causal(:,n_bands+1) = ZS;

%% Get all bursts
[tburst,ncycle,maxpeak,peaks,allpks] = DetectAllBursts(ZS_gamma,truth_thresh,Nconsc,mindist,endratio);
nburst = size(tburst,1);
disp(['Burst rate = ',num2str(1000*nburst/sum(seg(2,:)-seg(1,:))),' Hz']);

% Plot sample signal
figure(1);  hold on;
tshow = 700000:800000;
plot(tshow,ZS_gamma(tshow),'k');
plot(tshow,ZS(tshow),'Color',[0.5,0.5,0.5]);
for i = 1:nburst
    if tburst(i,2)>tshow(1) && tburst(i,1)<tshow(end)
        t = tburst(i,1):tburst(i,2);
        plot(t,ZS_gamma(t),'b');
        plot(peaks{i}(:,1),peaks{i}(:,2),'r.','MarkerSize',5);
        plot(maxpeak(i,1),maxpeak(i,2),'g.','Marker','^','MarkerSize',5);
    end
end
plot([tshow(1),tshow(end)],truth_thresh*[1,1],'y');

% Statistical analysis
figure(2);
subplot(411);
hist(ncycle,Nconsc:max(ncycle));
xlabel('number of cycles'); ylabel('incidence');
subplot(412);
plot(ncycle,maxpeak(:,2),'b.','MarkerSize',3);
axis tight;
xlabel('number of cycles'); ylabel('max peak value');
subplot(413);
intvlrange = 500;
intvl = tburst(2:end,1)-tburst(1:end-1,2);
hist(intvl(intvl<=intvlrange),1:intvlrange);
xlabel('interval between two bursts');  ylabel('incidence');
axis tight;
subplot(414);
hist(maxpeak(:,2),100);
xlabel('maximum peak value'); ylabel('incidence');
axis tight;

%% Spike triggered average of bursts
ivalid = any(bsxfun(@ge,tburst(:,1),tseg(1,:))&bsxfun(@le,tburst(:,2),tseg(2,:)),2);    % find bursts in valid range
nvalid = sum(ivalid);
showsize = 150;
colors = 'brygmck';

figure(3);
title('Spike trigger average of bursts');
ncyc = [5,10,max(ncycle)];
ncyc1 = [0,ncyc(1:end-1)];
mpeak = [0.5,0.7,max(maxpeak(:,2))];
mpeak1 = [truth_thresh,mpeak(1:end-1)];
% ntype = length(ncyc);
ntype = length(mpeak);
for j = 1:ntype;
    % id = find((ncycle>ncyc1(j)&ncycle<=ncyc(j))&ivalid);	% divide according to # of cycles
    id = find(maxpeak(:,2)>mpeak1(j)&maxpeak(:,2)<=mpeak(j)&ivalid);	% divide according to maximum peak value
    nv = numel(id);
    x_long = zeros(nv,showsize,n_bands+1);
    x_gamma = zeros(nv,showsize);
    for i = 1:nv
        x_long(i,:,:) = ZS_causal(tburst(id(i),1)-prelen-forward-1+(1:showsize),:);
        x_gamma(i,:) = ZS_gamma(tburst(id(i),1)-prelen-forward-1+(1:showsize));
    end
    x_avg = squeeze(mean(x_long,1));
    x_avg_gamma = mean(x_gamma,1);
    
    subplot(ntype,2,j*2-1);	hold on;
    plot(x_avg_gamma,colors(gamma_band));
    if j==ntype, legend('Non-causal filter gamma (64-84 Hz)');   end
    axis tight;	ylim([-0.8,0.8]);
    plot((prelen+forward+1)*[1,1],[-2,2],'y-');
    %ylabel(['# of cycles: ',num2str(ncyc1(j)),'-',num2str(ncyc(j))]);
    ylabel(['max peak: ',num2str(mpeak1(j)),'-',num2str(mpeak(j))]);
    subplot(ntype,2,j*2);	hold on;
    for i = 1:n_bands+1
        plot(x_avg(:,i),colors(i));
    end
    if j==ntype
        legend('delta(<5Hz)','theta(5-8Hz)','alpha(9-14Hz)','beta(15-40Hz)','gamma(64-84Hz)','high gamma(84-124Hz)','raw');
    end
    axis tight;	ylim([-1.5,1.5]);
    plot((prelen+forward+1)*[1,1],[-2,2],'y-');
end

% Spike triggered average of non-bursts
tstart = allpks(:,1)-prelen-forward;
tstop = tstart+showsize;
inonburst = true(size(tstart));     % find nonburst peaks
i = 1;	j = 1;
while j<=nburst && i<=length(inonburst)
    if tstop(i)>=tburst(j,1)
        inonburst(i) = false;
    end
    if tstart(i)>=tburst(j,2)
        j = j+1;
    end
    i = i+1;
end
inonburst = any(bsxfun(@ge,allpks(:,1),tseg(1,:))&bsxfun(@le,tstop,tseg(2,:)),2)&inonburst;  % find nonbursts in valid range
id = find(inonburst);
nnonburst = length(id);

% Plot non-burst peaks
% figure(1);
% for i = id'
%     if allpks(i,1)>tshow(1) && allpks(i,1)<tshow(end)
%         plot(allpks(i,1),allpks(i,2),'m.','MarkerSize',5);
%     end
% end

nnonburst = round(nnonburst*0.1);   % a subset of non-burst peaks (10%)
id = sort(randsample(id,nnonburst));
x_long = zeros(nnonburst,showsize,n_bands+1);
x_gamma = zeros(nnonburst,showsize);
for i = 1:nnonburst
    x_long(i,:,:) = ZS_causal(allpks(id(i),1)-prelen-forward-1+(1:showsize),:);
    x_gamma(i,:) = ZS_gamma(allpks(id(i),1)-prelen-forward-1+(1:showsize));
end
x_avg = squeeze(mean(x_long,1));
x_avg_gamma = mean(x_gamma,1);

figure(4);
title('Spike trigger average of non-bursts');
subplot(211);	hold on;
plot(x_avg_gamma,colors(gamma_band));
legend('Non-causal filter gamma (64-84 Hz)');
axis tight;	%ylim([-0.8,0.8]);
plot((prelen+forward+1)*[1,1],[-2,2],'y-');
subplot(212);	hold on;
for i = 1:n_bands+1
    plot(x_avg(:,i),colors(i));
end
legend('delta(<5Hz)','theta(5-8Hz)','alpha(9-14Hz)','beta(15-40Hz)','gamma(64-84Hz)','high gamma(84-124Hz)','raw');
axis tight;	%ylim([-1.5,1.5]);
plot((prelen+forward+1)*[1,1],[-2,2],'y-');

