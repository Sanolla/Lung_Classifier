function [freqResp, maxval_cut] = extractBreathingCycleFromSound(signal)
%        audiowrite('Data.wav',signal, 16000,'BitsPerSample',16);
 

fs = 16000;

signal = downsample(signal,4);
fs=fs/4;
L=20;
if length(signal)>L*fs
    signal_pre1 = signal(end-(L-1)*fs:end-fs);
else
    signal_pre1 = signal(1:end);
end
   
count = length(find(abs(signal_pre1)>0.99))/length(signal_pre1);


signal = schmidt_spike_removal(signal_pre1,fs,1.5,round(fs/10));


count2=1-length(find((signal_pre1-signal)==0))/length(signal_pre1);
% count = length(find(abs(signal_pre1)>0.99))/length(signal_pre1);

% figure; plot(signal_pre1); hold; plot(signal,'r');
if count>0.015 
    d = single(firpm(25,[0 0.9],[1 1],'hilbert'));
%     freqResp=1000;
    flag=0;
else
%     signal = butterworth_low_pass_filter(signal,2,60,fs);
    flag=1;
    d = single(firpm(25,[0 1],[1 1],'hilbert'));
end
    signalf_pre=filter(d,1,signal);
%     figure; plot(signalf_pre);

    signalf = schmidt_spike_removal(signalf_pre,fs,3,round(fs/10));
    
%      figure; subplot(211); plot(signal);hold; plot(signalf_pre,'r');
     
%     [pxx,f] = pspectrum(signalf,4000);
    
%     subplot(212);
%     plot(f,pow2db(pxx))
%     grid on
%     xlabel('Frequency (Hz)')
%     ylabel('Power Spectrum (dB)')
%     title('Default Frequency Resolution')
    
    count3=1-length(find((signalf_pre-signalf)==0))/length(signalf_pre);
%     signalf = signalf(50:end);
    %     figure; plot(signalf);

    %     figure; plot(homomorphic_envelope_for_lung(500:end));
    %     homomorphic_envelope_for_lung = signalf;
%     homomorphic_envelope_for_lung = medfilt1(signalf,101); %.*sign(signalf);
%       homomorphic_envelope_for_IsBreathing = Homomorphic_Envelope_with_Hilbert(signalf,fs,8);
      homomorphic_envelope_for_lung = Homomorphic_Envelope_with_Hilbert(signalf,fs,8);
%     homomorphic_envelope_for_lung=homomorphic_envelope_for_lung(3000:end);

   
[maxfreq_cut, maxval_cut] = findRespFreq(homomorphic_envelope_for_lung(500:end));


% if maxval_cut>0.00001&&flag==1
    freqResp=maxfreq_cut*60;
% else
%     freqResp=0;
% end


    if freqResp>50
        freqResp=freqResp/2;
    end
    


%         figure; plot(signalf(500:end)); hold; plot(homomorphic_envelope_for_lung(500:end) ,'r');



end
