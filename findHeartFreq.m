function [maxfreq_cut_heart, maxval_cut_heart]= findHeartFreq(homomorphic_envelope_for_heart)

 nfft = 2^20;
    f = 0:4000/nfft:2000-1/nfft;
    high_cutoffidx_heart = length(find(f <= 1.7));
    low_cutoffidx_heart = find(f > 0.9,1);
       
    f_cut_heart = f(low_cutoffidx_heart:high_cutoffidx_heart);
    
    datafft_for_heart = fft(homomorphic_envelope_for_heart-mean(homomorphic_envelope_for_heart),nfft);
    
    ps_heart = abs(datafft_for_heart).^2;
    ps_heart = ps_heart(1:round(length(ps_heart)/2));
    ps_heart = ps_heart/sum(ps_heart);
   
    % keep only the non-redundant portion
    ps_cut_heart = ps_heart(low_cutoffidx_heart:high_cutoffidx_heart);
%     
%  figure; plot(f(1:1000),ps_heart(1:1000));
%     
    [PKS_heart,LOCS_heart,W] = findpeaks(ps_cut_heart);
        
    [~,Ind_heart] = max(PKS_heart.*W);
    maxfreq_cut_heart = f_cut_heart(LOCS_heart(Ind_heart));
    maxval_cut_heart = ps_cut_heart(LOCS_heart(Ind_heart));
    
%     if maxval_cut_heart<0.002
%         maxfreq_cut_heart=0;
%     end
    
    