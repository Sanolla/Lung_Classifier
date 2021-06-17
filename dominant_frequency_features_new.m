function [maxfreq, maxval, maxratio, sumrange,spectralEntropy] = dominant_frequency_features_new(datafft, fs, cutoff) %#codegen
% This function estimates the dominant frequency of each channel (COLUMN) of the
% input data.  First, it calculates the power spectrum of the signal.  Then it finds the 
% maximum frequency over as well as the "energy" at that point; options below allow normalization
% by the total or mean energy.  Note this does not restrict the range of frequencies to search 
% for a maximum.
% 
% data         matrix containing data where each COLUMN corresponds to a channel
% fs           sampling rate (Hz)
% cutoff       cutoff frequency (Hz)
% maxfreq      frequency at which the max of the spectrum occurs (Hz) (ROW VECTOR)
% maxratio     ratio of the energy of the maximum to the total energy (ROW VECTOR)
%Copyright (c) 2016, MathWorks, Inc. 
maxval=[0,0,0,0,0];
maxfreq=[0,0,0,0,0];
maxratio=[0,0,0,0,0];
sumrange=[0,0,0,0,0];

f = fs/2*linspace(0,1,length(datafft)/2);
cutoffidx = length(find(f <= cutoff));
f = f(1:cutoffidx);

% calculate the power spectrum using FFT method
ps = abs(datafft/length(datafft)); 
 

% keep only the non-redundant portion
ps = ps(1:cutoffidx);

h = histcounts(ps,150,'Normalization', 'probability')+10e-8;
spectralEntropy = -sum(h.*log2(h));


% locate max value below cutoff
[maxval(1), maxind] = max(ps(1:round(cutoffidx/5)));
maxfreq(1) = f(maxind);
% calculate peak energy by summing energy from maxfreq-delta to maxfreq+delta
% then normalize by total energy below cutoff
delta = 5;  % Hz
maxratio(1) = 0;
maxrange = f>=maxfreq(1)-delta & f<=maxfreq(1)+delta;
maxratio(1) = sum(ps(maxrange));
sumrange(1)=sum(ps(1:round(cutoffidx/5)));

[maxval(2), maxind] = max(ps(round(cutoffidx/5)+1:round(2*cutoffidx/5)));
maxfreq(2) = f(maxind+round(cutoffidx/5));
% calculate peak energy by summing energy from maxfreq-delta to maxfreq+delta
% then normalize by total energy below cutoff
delta = 5;  % Hz
maxratio(2) = 0;
maxrange = f>=maxfreq(2)-delta & f<=maxfreq(2)+delta;
maxratio(2) = sum(ps(maxrange));
sumrange(2)=sum(ps(round(cutoffidx/5)+1:round(2*cutoffidx/5)));

[maxval(3), maxind] = max(ps(round(2*cutoffidx/5)+1:round(3*cutoffidx/5)));
maxfreq(3) = f(maxind+round(2*cutoffidx/5));
% calculate peak energy by summing energy from maxfreq-delta to maxfreq+delta
% then normalize by total energy below cutoff
delta = 5;  % Hz
maxratio(3) = 0;
maxrange = f>=maxfreq(3)-delta & f<=maxfreq(3)+delta;
maxratio(3) = sum(ps(maxrange));
sumrange(3)=sum(ps(round(2*cutoffidx/5)+1:round(3*cutoffidx/5)));

[maxval(4), maxind] = max(ps(round(3*cutoffidx/5)+1:round(4*cutoffidx/5)));
maxfreq(4) = f(maxind+round(3*cutoffidx/5));
% calculate peak energy by summing energy from maxfreq-delta to maxfreq+delta
% then normalize by total energy below cutoff
delta = 5;  % Hz
maxratio(4) = 0;
maxrange = f>=maxfreq(4)-delta & f<=maxfreq(4)+delta;
maxratio(4) = sum(ps(maxrange));
sumrange(4)=sum(ps(round(3*cutoffidx/5)+1:round(4*cutoffidx/5)));

[maxval(5), maxind] = max(ps(round(4*cutoffidx/5)+1:end-5));
maxfreq(5) = f(maxind+round(4*cutoffidx/5));
% calculate peak energy by summing energy from maxfreq-delta to maxfreq+delta
% then normalize by total energy below cutoff
delta = 5;  % Hz
maxratio(5) = 0;
maxrange = f>=maxfreq(5)-delta & f<=maxfreq(5)+delta;
maxratio(5) = sum(ps(maxrange));
sumrange(5)=sum(ps(round(4*cutoffidx/5)+1:cutoffidx));


