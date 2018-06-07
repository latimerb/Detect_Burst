function [ tburst,ncycle,maxpeak,peaks,allpeaks ] = DetectAllBursts( X,thresh,Nconsc,mindist,endratio )
% mindist = 10;       % Minimum interval between peaks (10 ms)
% thresh = 0.352;     % Detect amplitude threshold
% Nconsc = 3;         % Number of consecutive peaks(>threshold) required (must be >=2)
% endratio = 0.25;    % Minimum ratio of the last peak value to the maximum peak value in a burst

% X is the input signal where bursts are to be detected.
% Let N be the total number of bursts detected.
% tburst is an N*2 matrix. The first and the second columns are the time of
% the first and the last peak in each burst respectively.
% ncycle is an N*1 vector with the number of cycles of corresponding burst.
% maxpeak is an N*2 matrix marking the maximum peak in each burst. The first
% column are the time of the maximum peaks. The second column are their values.
% peaks is an N*1 cell matrix, each cell of which is an M*2 matrix marking
% all the peaks in each burst, where M is the number of cycles in each burst.
% allpeaks is a matrix marking the time and value of all valid peaks in the whole signal.

T = numel(X);
X = reshape(X,T,1);	% Filtered signal in detection window
d2X = [0;-diff(sign(diff(X)))];     % 2 or [1,1] => peak, -2 or [-1,-1] => trough
tpk = find(d2X==2 | d2X==1&[d2X(2:end);0]==1);	% Peak timing

% Eliminate peaks with interval<mindist
npk = length(tpk);  % number of peaks
vpk = true(npk,1);	% valid peaks
intvl = [mindist+1;diff(tpk);mindist+1];
[~,ipk] = sort(X(tpk),'descend');
for i = 1:npk
    id = ipk(i);
    if vpk(id)
        pi = id-1;
        dist = intvl(id);
        while dist<mindist && vpk(pi)
            vpk(pi) = false;
            dist = dist+intvl(pi);
            pi = pi-1;
        end
        pi = id+1;
        dist = intvl(pi);
        while dist<mindist && vpk(pi)
            vpk(pi) = false;
            pi = pi+1;
            dist = dist+intvl(pi);
        end
    end
end
tpk = tpk(vpk);     % time of all valid peaks
npk = length(tpk);  % number of valid peaks

% Check whether Nconsc consecutive peaks are over threshold
burst = zeros(floor(T/mindist/Nconsc),5);
overthresh = false;
nburst = 0;
for i = 1:npk
    t = tpk(i);
    x = X(t);
    if ~overthresh
        if x > thresh
            overthresh = true;
            start = i;
            ncyc = 1;
            maxpk = x;
            tmax = t;
        end
    else
        if x > maxpk
            maxpk = x;
            tmax = t;
            ncyc = ncyc+1;
            stop = i;
        else
            if ncyc < Nconsc
                if x > thresh
                    ncyc = ncyc+1;
                    stop = i;
                else
                    overthresh = false;
                end
            else
                if x >= maxpk*endratio
                    ncyc = ncyc+1;
                    stop = i;
                else
                    nburst = nburst+1;
                    burst(nburst,:) = [start,stop,ncyc,tmax,maxpk];
                    overthresh = false;
                end
            end
        end
    end
end
tburst = [tpk(burst(1:nburst,1)),tpk(burst(1:nburst,2))];
ncycle = burst(1:nburst,3);
maxpeak = burst(1:nburst,4:5);

if nargout>=4
    peaks = cell(nburst,1);
    for i = 1:nburst
        t = tpk(burst(i,1):burst(i,2));
        peaks{i} = [t,X(t)];
    end
end
if nargout>=5
    allpeaks = [tpk,X(tpk)];
end

end