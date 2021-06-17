function [maxfreq_cut, maxval_cut] = findRespFreq(homomorphic_envelope_for_lung)

nfft = 2^nextpow2(2*length(homomorphic_envelope_for_lung)); %2^16;
    f = 0:4000/nfft:2000-1/nfft;
    high_cutoffidx = length(find(f <= 0.78));
    low_cutoffidx = find(f > 0.1,1);
    
    f_cut = f(low_cutoffidx:high_cutoffidx);
    
    datafft_for_lung = fft(homomorphic_envelope_for_lung-mean(homomorphic_envelope_for_lung),nfft);
    
    ps_lung = abs(datafft_for_lung/nfft);
    ps_lung = ps_lung(1:round(length(ps_lung)/2));
%         ps_lung = ps_lung(1:40);

%     ps_lung = ps_lung/sum(ps_lung);
    
   
%     % keep only the non-redundant portion
    ps_cut_lung = ps_lung(low_cutoffidx:high_cutoffidx);

%      figure; plot(f(1:1000),ps_lung(1:1000));
     if length(ps_cut_lung)<3
         maxfreq_cut = 0;
         maxval_cut = 0; 
     else
        [PKS,LOCS,W] = findpeaks(ps_cut_lung);
     

%     if size(LOCS,2)*size(LOCS,1)==0
%         high_cutoffidx = length(find(f <= 1.4));
%         low_cutoffidx = find(f > 0.13,1);
%         f_cut = f(low_cutoffidx:high_cutoffidx);
%         ps_cut_lung = ps_lung(low_cutoffidx:high_cutoffidx,:);
%         [PKS,LOCS] = findpeaks(ps_cut_lung);
%     else
%         if LOCS(1)==1
%             PKS = PKS(2:end);
%             LOCS = LOCS(2:end);
%             
%         end
%     end

if size(PKS,1)*size(PKS,2)==0
    maxfreq_cut = 0;
    maxval_cut = 0; 
else
    [~,Ind] = max(PKS.*W);
    maxfreq_cut = f_cut(LOCS(Ind));
    maxval_cut = ps_cut_lung(LOCS(Ind)); 
end
     end
end
    