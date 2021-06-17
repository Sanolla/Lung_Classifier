function [feature_table_all,old_name,position] = extractFeatures_win16(fds,window_length,window_overlap,old_name,position)
%Function to extract only features for heart sound classification demo.
%Copyright (c) 2017, MathWorks, Inc. 

warning off

overlap_length = window_length * window_overlap / 100;
step_length = window_length - overlap_length;

feature_table_all = table();

labelMap = containers.Map('KeyType','int32','ValueType','char');
keySet = {-1, 1, 2, 3, 4, 5, 6, 7};

valueSet = {'Normal','COPD','Pneumonia', 'COVID19', 'Heart','Asthma','BR','asy' };

labelMap = containers.Map(keySet,valueSet);


count_bad = 0;

for i_file=1:length(fds.Files)
%     [~, signal] = AdaptiveFilterDSP_Noa(fds.Files{i_file});
    
    fid = fopen(fds.Files{i_file},'rb');
    [temp, ~] = fread(fid,'int16');
         fclose(fid);
   split_for_name = split(fds.Files{i_file},'\'); 

    if contains(fds.Files{i_file},'test')
        flag_test=1;
        name = split_for_name{end-1};
    else
        flag_test=0;
        name = split_for_name{end-2};
    end
    
    if strcmp(string(name),old_name)
        position=position+1;
    else
        position=1;
    end
    
    if (contains(fds.Files{i_file}, '83000')||contains(fds.Files{i_file}, '84000'))==0
    
%     if (position==8||position==9||position==11||position==12)==0
%         old_name = string(name);
%         continue;
%     end
    
    if position<8
        old_name = string(name);
        continue;
    end
    
    end
    
    signal = temp(1:2:end)/(2^15);
%     x1 = temp(1:2:end)/(2^15);
    x2 = temp(2:2:end)/(2^15);
      
    
    fs = 16000;
%     signal = WienerScalart(x1, fs);

%     signal = LMS_SE_with_diff(x1,x2); % converge to body lowpass filter


    old=0;
    if contains(fds.Files{i_file}, '.wav')
        continue;
    elseif contains(fds.Files{i_file}, 'xml')
        continue;
    elseif contains(fds.Files{i_file}, 'Healthy')
        current_class = -1;
        %         continue;
    elseif contains(fds.Files{i_file}, 'COPD')
        current_class=1;
        %         continue;
    elseif contains(fds.Files{i_file}, 'Pneumonia')
        current_class=2;
        %         continue;
    elseif contains(fds.Files{i_file}, 'Asthma')
        current_class=5;
        %          continue;
    elseif contains(fds.Files{i_file}, 'Corona')
        current_class=3;
        %         continue;
        % elseif contains(fds.Files{i_file}, 'BR')
        %         current_class=3;
        % %         continue;
        elseif contains(fds.Files{i_file}, 'asy')
                current_class=7;
        %         continue;
        %     elseif contains(fds.Files{i_file}, '1111000')
        %         signal = signal*2;
        %         continue;
    end
    
%      if contains(fds.Files{i_file}, '1111000')
%         signal = signal*2;
%      end
    
%     if contains(fds.Files{i_file}, 'old')
%         old=1;
%         continue;
%     end
    
    

    
%     name = [split_for_name{end-2} '_' split_for_name{end}];
    
%     if strcmp(string(name),'74000006')
%        name;
%     end
    
    
    
%     if contains(fds.Files{i_file},'__8')
%         position=8;
%     elseif contains(fds.Files{i_file},'__9')
%         position=9;
%     elseif contains(fds.Files{i_file},'__11')
%         position=11;
%     elseif contains(fds.Files{i_file},'__12')
%         position=12;
%     end
    
%     katerinaQ=0;
%    I_name= strfind(out_table(:,1)',str2double(name(1:end-9)));
%    if position<15
%         katerinaQ=min(out_table(I_name,position+1));
%    if min(out_table(I_name,position+1))==3
%      position  
%           playsignal = audioplayer(signal, 16000);
%           play(playsignal); 
%        continue;
%    end
%    else
%        
%    end
       
        
%      signal = butterworth_high_pass_filter(signal,9,80,fs);
%      signal = highpass(signal,160,fs,'ImpulseResponse','iir');
%      signal = highpass(signal,160,16000,'ImpulseResponse','iir');
%      signal = highpass(signal,160,16000,'ImpulseResponse','fir');
% %      f=fs/2*[0:length(signalfilt)-1]/length(signalfilt);
%      figure; plot(f,abs(fft(signal)));hold; plot(f,abs(fft(signalfilt)),'r'); 
     
%      signal = signal(100:end-4000);
% figure;plot(fsignal);
% figure; plot(signal); hold; plot(dsignal,'r');

%      dsignal = decimate_Itai(signal);
%      dx2 = decimate_Itai(x2);

%      dsignal = downsample(signal,4);
%      dx2 = downsample(x2,4);
%      fs=fs/4;
     dsignal = signal;
     dx2 = x2;
     NN=length(dsignal);

     ind_begin_vec=find(abs(diff(dsignal(1:fs)))>0.005);
     ind_end_vec = find(abs(diff(dsignal(end-fs:end)))>0.005);
     
    
     if size(ind_begin_vec,1)==0
         ind_begin=1;
     else
     ind_begin = ind_begin_vec(end)+1;
     end
     if size(ind_end_vec,1)==0
        ind_end=length(dsignal);
     else
         ind_end = length(dsignal)-fs+ind_end_vec(1)-1;
     end
     
     

%     figure; plot(dsignal);hold; plot(ind_begin,dsignal(ind_begin),'*r'); plot(ind_end,dsignal(ind_end),'*r');

     L1=length(dsignal)/fs;
%      sig_before_cut = dsignal;
     dsignal = dsignal(ind_begin:ind_end-1);
     dx2 = dx2(ind_begin:ind_end-1);

     L2=length(dsignal)/fs;
     
     if L2<2
%          if flag_test==1
% figure; plot(sig_before_cut);
% figure; plot(sig_before_cut);hold; plot(ind_begin,sig_before_cut(ind_begin),'*r'); plot(ind_end,sig_before_cut(ind_end),'*r');
%           playsignal = audioplayer(dsignal, 4000);
%           play(playsignal); 
%          end
         continue;
     end
     
     L=12;
     if length(dsignal)>L*fs
        dsignal = dsignal(end-L*fs:end);
        dx2 = dx2(end-L*fs:end);
     end
     
    clear new_temp
    new_temp(:,1) = dsignal;
    new_temp(:,2) = dx2;
    
%  if (contains(fds.Files{i_file}, '76000')||contains(fds.Files{i_file}, '75000'))==0
      
signal = noise_reduction_AuxIVA(new_temp);
dsignal = signal(:,1,1);

% else
%       signal = temp(1:2:end)/(2^15);
% end
     
%     [ ~, isValid ] = IsGoodQualityFile( fds.Files{i_file} );
%     if isValid==0
%         continue;
%     end


%      dsignal = dsignal/max(abs(dsignal));
%           dsignal = dsignal/quantile(abs(dsignal), 0.90);

     

%      L=10;
%      if length(dsignal)>L*fs
%         dsignal = dsignal(end-(L-1)*fs-1:end-fs/2);
%      else
% %          continue;
%         dsignal = dsignal(fs:end-fs/2);
%      end
%      figure; plot(dsignal);
     
%     count_move = length(find(abs(dsignal)>0.99))/length(dsignal);
%     if count_move>0.01
%         continue;
%     end

%     [RespRate, val_breathing] = extractBreathingCycleFromSound(dsignal);
%     [HeartRate, val_heart] = extractHeartCycleFromSound(dsignal);
%    
%     signal = schmidt_spike_removal(dsignal,fs,1.5,round(fs/10));
        
%     window_length = length(signal)/fs;
    

%     f_LP=1999; %fs/2;
%     sig_amp = max(abs(signal));

%     d = single(firpm(60,[0 1],[1 1],'hilbert'));
% %     
%      dsignal=filter(d,1,dsignal);
%      dsignal = dsignal(fs/8:end-fs/8);



%      
%     signal = signal(10:end);
%     figure; plot(current_signal); hold; plot(signalf,'r')
%     signalf = schmidt_spike_removal(signalf,fs,1.5,round(fs/10));
%     sig_ampf = max(abs(signalf));
    

% if contains(fds.Files{i_file}, '71000001')
%    signal = signal;
% end

   homomorphic_envelope_for_lung = Homomorphic_Envelope_with_Hilbert(dsignal,fs,5);
%    homomorphic_envelope_for_lung = homomorphic_envelope_for_lung(500:end-500);
    Env=sum(homomorphic_envelope_for_lung)/(length(homomorphic_envelope_for_lung));
% figure; plot(homomorphic_envelope_for_lung);

% f = 4000/2*linspace(0,1,length(homomorphic_envelope_for_lung)/2);
% envfft = fft(homomorphic_envelope_for_lung-mean(homomorphic_envelope_for_lung));
% figure; plot(f,abs(envfft(1:length(f))));

feature_table = table();
% feature_table.old=old;
% feature_table.SurrNoise=surr_noise;
% feature_table.maxval_ratio = maxval_cut/maxval;

% if size(val_breathing,1)==0
%     val_breathing=0;
% end
% if size(val_heart,1)==0
%     val_heart=0;
% end
% if size(HeartRate,2)==0
%     HeartRate=0;
% end

N = length(dsignal); % Bernhard: to keep track of #samples in files we process

% number_of_windows = floor( (N - overlap_length*fs) / (fs * step_length));

number_of_windows=1;
window_length = length(dsignal)/fs;
% if number_of_windows==0
%     flag_too_short=1;
%     number_of_windows=1;
% else
%     flag_too_short=0;
% end

 for iwin = 1:number_of_windows
%         if iwin>1
%             continue;
%         end
%      if flag_too_short==1
%         current_signal = interp1(1:length(dsignal),dsignal,1:window_length * fs,'next','extrap');
% %         figure; plot(current_signal);
%       current_signal = current_signal';
% 
%      else
        current_start_sample = (iwin - 1) * fs * step_length + 1;
        current_end_sample = current_start_sample + window_length * fs - 1;
        current_signal = dsignal(current_start_sample:current_end_sample);
%      end
%         feature_table.N(iwin, 1) = NN/fs;
%         feature_table.ValBreathing(iwin, 1) = val_breathing;
%         feature_table.ValHeart(iwin, 1) = val_heart;
% 
%         feature_table.HeartRate(iwin, 1) = HeartRate;
%         feature_table.RespRate(iwin, 1) = RespRate;
%         feature_table.SigAmp(iwin, 1) = sig_amp;
%         feature_table.SigAmpf(iwin, 1) = sig_ampf;

 
       %[homomorphic_envelope,lo] = envelope(signal_for_env,5000,'rms'); %For fs=16000
        feature_table.L1(iwin, 1)=L1;
        feature_table.L2(iwin, 1)=L2;
        
        feature_table.cut_begin(iwin, 1)=(ind_begin-1)/fs;
        feature_table.cut_end(iwin, 1)=L1-ind_end/fs;

        feature_table.Env(iwin, 1)=Env;
        feature_table.name(iwin, 1)=string(name);
        feature_table.position(iwin, 1)=position;
        old_name = string(name);
       
        % Calculate mean value feature
        feature_table.meanValue(iwin, 1) = mean(abs(current_signal));
        
        % Calculate median value feature
        feature_table.medianValue(iwin, 1) = median(abs(current_signal));

        % Calculate standard deviation feature
        feature_table.standardDeviation(iwin, 1) = std(current_signal);
        feature_table.standardDeviationEnv(iwin, 1) = std(homomorphic_envelope_for_lung);
        feature_table.meanEnv(iwin, 1) = mean(homomorphic_envelope_for_lung);

        % Calculate mean absolute deviation feature
        feature_table.meanAbsoluteDeviation(iwin, 1) = mad(current_signal);
        
        % Calculate signal 25th percentile feature
        feature_table.quantile25(iwin, 1) = quantile(abs(current_signal), 0.25);

        % Calculate signal 75th percentile feature
        feature_table.quantile75(iwin, 1) = quantile(abs(current_signal), 0.75);

        % Calculate signal inter quartile range feature
        feature_table.signalIQR(iwin, 1) = iqr(abs(current_signal));
        
        % Calculate skewness of the signal values
        feature_table.sampleSkewness(iwin, 1) = skewness(current_signal);

        % Calculate kurtosis of the signal values
        feature_table.sampleKurtosis(iwin, 1) = kurtosis(current_signal);

        % Calculate Shannon's entropy value of the signal
        h = histcounts(current_signal,150,'Normalization', 'probability')+10e-9;
        entropy = -sum(h.*log2(h));
        feature_table.signalEntropy(iwin, 1) = entropy; 

        % Calculate spectral entropy of the signal
        feature_table.spectralEntropy(iwin, 1) = spectral_entropy(current_signal, fs, fs/8);

        % Extract features from the power spectrum
        [maxfreq, maxval, maxratio, sumrange] = dominant_frequency_features(current_signal, fs, fs/8);
        feature_table.dominantFrequencyValue(iwin, 1) = maxfreq(1);
        feature_table.dominantFrequencyMagnitude(iwin, 1) = maxval(1);
        feature_table.dominantFrequencyRatio(iwin, 1) = maxratio(1);
        feature_table.sumrange(iwin, 1) = sumrange(1);
        feature_table.dominantFrequencyValue2(iwin, 1) = maxfreq(2);
        feature_table.dominantFrequencyMagnitude2(iwin, 1) = maxval(2);
        feature_table.dominantFrequencyRatio2(iwin, 1) = maxratio(2);
        feature_table.sumrange2(iwin, 1) = sumrange(2);

        feature_table.dominantFrequencyValue3(iwin, 1) = maxfreq(3);
        feature_table.dominantFrequencyMagnitude3(iwin, 1) = maxval(3);
        feature_table.dominantFrequencyRatio3(iwin, 1) = maxratio(3);
        feature_table.sumrange3(iwin, 1) = sumrange(3);

        feature_table.dominantFrequencyValue4(iwin, 1) = maxfreq(4);
        feature_table.dominantFrequencyMagnitude4(iwin, 1) = maxval(4);
        feature_table.dominantFrequencyRatio4(iwin, 1) = maxratio(4);
        feature_table.sumrange4(iwin, 1) = sumrange(4);

        feature_table.dominantFrequencyValue5(iwin, 1) = maxfreq(5);
        feature_table.dominantFrequencyMagnitude5(iwin, 1) = maxval(5);
        feature_table.dominantFrequencyRatio5(iwin, 1) = maxratio(5);
        feature_table.sumrange5(iwin, 1) = sumrange(5);

        % Extract wavelet features
        % REMOVED because didn't contribute much to final model (only 1 of
        % them was selected by NCA among the "important" features)
        
        % Extract Mel-frequency cepstral coefficients
        Tw = window_length*1000;                % analysis frame duration (ms)
        Ts = 10;                % analysis frame shift (ms)
        alpha = 0.97;           % preemphasis coefficient
        M = 20;                 % number of filterbank channels 
        C = 19;                 % number of cepstral coefficients
        L = 22;                 % cepstral sine lifter parameter
        LF = 0;                 % lower frequency limit (Hz)
        HF = fs/8;               % upper frequency limit (Hz)

        [MFCCs, ~, Qf, ~] = mfcc(current_signal, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L);
        feature_table.MFCC1(iwin, 1) = MFCCs(1);
        feature_table.MFCC2(iwin, 1) = MFCCs(2);
        feature_table.MFCC3(iwin, 1) = MFCCs(3);
        feature_table.MFCC4(iwin, 1) = MFCCs(4);
        feature_table.MFCC5(iwin, 1) = MFCCs(5);
        feature_table.MFCC6(iwin, 1) = MFCCs(6);
        feature_table.MFCC7(iwin, 1) = MFCCs(7);
        feature_table.MFCC8(iwin, 1) = MFCCs(8);
        feature_table.MFCC9(iwin, 1) = MFCCs(9);
        feature_table.MFCC10(iwin, 1) = MFCCs(10);
        feature_table.MFCC11(iwin, 1) = MFCCs(11);
        feature_table.MFCC12(iwin, 1) = MFCCs(12);
        feature_table.MFCC13(iwin, 1) = MFCCs(13);
        feature_table.MFCC14(iwin, 1) = MFCCs(14);
        feature_table.MFCC15(iwin, 1) = MFCCs(15);
        feature_table.MFCC16(iwin, 1) = MFCCs(16);
        feature_table.MFCC17(iwin, 1) = MFCCs(17);
        feature_table.MFCC18(iwin, 1) = MFCCs(18);
        feature_table.MFCC19(iwin, 1) = MFCCs(19);
        feature_table.MFCC20(iwin, 1) = MFCCs(20);
        feature_table.Qf(iwin, 1) = Qf;

         [MFCCs, ~, Qf, ~] = mfcc(current_signal, fs, Tw, Ts, alpha, @hamming, [LF 20], M, C+1, L);
        feature_table.MFCC1infra(iwin, 1) = MFCCs(1);
        feature_table.MFCC2infra(iwin, 1) = MFCCs(2);
        feature_table.MFCC3infra(iwin, 1) = MFCCs(3);
        feature_table.MFCC4infra(iwin, 1) = MFCCs(4);
        feature_table.MFCC5infra(iwin, 1) = MFCCs(5);
        feature_table.MFCC6infra(iwin, 1) = MFCCs(6);
        feature_table.MFCC7infra(iwin, 1) = MFCCs(7);
        feature_table.MFCC8infra(iwin, 1) = MFCCs(8);
        feature_table.MFCC9infra(iwin, 1) = MFCCs(9);
        feature_table.MFCC10infra(iwin, 1) = MFCCs(10);
        feature_table.MFCC11infra(iwin, 1) = MFCCs(11);
        feature_table.MFCC12infra(iwin, 1) = MFCCs(12);
        feature_table.MFCC13infra(iwin, 1) = MFCCs(13);
        feature_table.MFCC14infra(iwin, 1) = MFCCs(14);
        feature_table.MFCC15infra(iwin, 1) = MFCCs(15);
        feature_table.MFCC16infra(iwin, 1) = MFCCs(16);
        feature_table.MFCC17infra(iwin, 1) = MFCCs(17);
        feature_table.MFCC18infra(iwin, 1) = MFCCs(18);
        feature_table.MFCC19infra(iwin, 1) = MFCCs(19);
        feature_table.MFCC20infra(iwin, 1) = MFCCs(20);
        feature_table.Qfinfra(iwin, 1) = Qf;
                       
        % Assign class label to the observation
     if iwin == 1
            
                feature_table.class = {labelMap(current_class)};
%                 feature_table.class = current_class;
            
     else
            
                feature_table.class{iwin, :} = labelMap(current_class);
%                feature_table.class{iwin-bad_quality, 1} = current_class;

     end 
     
     if contains(fds.Files{i_file}, '3331')
         if feature_table.position==1
             feature_table.position=11;
         elseif feature_table.position==2
             feature_table.position=8;
         elseif feature_table.position==3
             feature_table.position=12;
         elseif feature_table.position==4
             feature_table.position=9; 
         end
     end
     
     if contains(fds.Files{i_file}, '4441')
         if feature_table.position==1
             feature_table.position=8;
         elseif feature_table.position==2
             feature_table.position=9;
         elseif feature_table.position==3
             feature_table.position=11;
         elseif feature_table.position==4
             feature_table.position=12; 
         end
     end
     
      
     if contains(fds.Files{i_file},'MADA')
         if feature_table.position==1
             feature_table.position=11;
         elseif feature_table.position==2
             feature_table.position=8;
         elseif feature_table.position==3
             feature_table.position=12;
         elseif feature_table.position==4
             feature_table.position=9;
         end
     end
 end
    
      feature_table_all = [feature_table_all; feature_table];
end    
    
    