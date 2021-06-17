function [HeartRate, maxval_cut_heart] = extractHeartCycleFromSound(signal)
%        audiowrite('Data.wav',signal, 16000,'BitsPerSample',16);


fs = 16000;

signal = downsample(signal,4);
fs=fs/4;
if length(signal)>18*fs
    signal_pre1 = signal(end-17*fs:end-fs);
else
    signal_pre1 = signal(1:end);
end
count = length(find(abs(signal_pre1)>0.99))/length(signal_pre1);

signal = schmidt_spike_removal(signal_pre1,fs,1.5,round(fs/10));

% figure; plot(signalf);
if count>1
    HeartRate=0;
    maxval_cut_heart=0;
else
    

    
    d = single(firpm(20,[0 1],[1 1],'hilbert'));
    
    signalf=filter(d,1,signal);
    signalf = schmidt_spike_removal(signalf,fs,1.5,round(fs/10));
     homomorphic_envelope_for_heart = Homomorphic_Envelope_with_Hilbert(signalf, fs,8);

%     homomorphic_envelope_for_heart = medfilt1(abs(signalf),5);
%     audio_data = butterworth_low_pass_filter(signalf,2,400,fs);


%     homomorphic_envelope_for_heart(signalf>=0) = medfilt1(signalf(signalf>=0),5); %.*sign(signalf);
%     homomorphic_envelope_for_heart(signalf<0) = medfilt1(signalf(signalf<0),5); %.*sign(signalf);
%     [homomorphic_envelope_for_heart,~] = envelope(homomorphic_envelope_for_heart,500,'peak');
%     homomorphic_envelope_for_heart = homomorphic_envelope_for_heart(200:end-200);
%     
    %   figure; plot(signal); hold; plot(homomorphic_envelope_for_heart ,'r');
    
    
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [maxfreq_cut_heart, maxval_cut_heart] = findHeartFreq(homomorphic_envelope_for_heart);
    HeartRate = maxfreq_cut_heart*60;
end
end
