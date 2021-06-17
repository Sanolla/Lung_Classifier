function [CC,CC_infra,Q,Q_infra] = mfcc_with_infra( datafft, fs, Tw, Ts, R, M, N, L ) %#codegen

    datafft = datafft * 2^15;

%     Nw = round( 1E-3*Tw*fs );    % frame duration (samples)
%     Ns = round( 1E-3*Ts*fs );    % frame shift (samples)
% 
%     nfft = 2^nextpow2( Nw );     % length of FFT analysis 
    K = length(datafft)/2+1;                % length of the unique part of the FFT 


    %% HANDY INLINE FUNCTION HANDLES

    % Forward and backward mel frequency warping (see Eq. (5.13) on p.76 of [1]) 
    % Note that base 10 is used in [1], while base e is used here and in HTK code
    hz2mel = @( hz )( 1127*log(1+hz/700) );     % Hertz to mel warping function
    mel2hz = @( mel )( 700*exp(mel/1127)-700 ); % mel to Hertz warping function

    % Type III DCT matrix routine (see Eq. (5.14) on p.77 of [1])
    dctm = @( N, M )( sqrt(2.0/M) * cos( repmat([0:N-1].',1,M) ...
                                       .* repmat(pi*([1:M]-0.5)/M,N,1) ) );

    % Cepstral lifter routine (see Eq. (5.12) on p.75 of [1])
    ceplifter = @( N, L )( 1+0.5*L*sin(pi*[0:N-1]/L) );


    %% FEATURE EXTRACTION 

    % Preemphasis filtering (see Eq. (5.1) on p.73 of [1])
%     speech = filter( [1 -alpha], 1, speech ); % fvtool( [1 -alpha], 1 );
    
    % Framing and windowing (frames as columns)
%     frames = vec2frames( speech, Nw, Ns, 'cols', window, false );

    % Magnitude spectrum computation (as column vectors)
    MAG = abs(datafft)/length(datafft);
    f=linspace(0,fs/2,length(datafft)/2+1);
   
    ind_f = length(find(f<R(2)));
    Q=sum(MAG(1:ind_f));
    ind_f_infra = length(find(f<20));
    Q_infra=sum(MAG(1:ind_f_infra));
    
%     homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(MAG, fs,5);
    % Triangular filterbank with uniformly spaced filters on mel scale
    H = trifbank( M, K, R, fs, hz2mel, mel2hz ); % size of H is M x K 
    H_infra = trifbank( M, K, [0,20], fs, hz2mel, mel2hz ); % size of H is M x K 

    % Filterbank application to unique part of the magnitude spectrum
%     FBE = (fs/16000)*H * MAG(1:K,:); % FBE( FBE<1.0 ) = 1.0; % apply mel floor
    FBE = H * MAG(1:K,:); % FBE( FBE<1.0 ) = 1.0; % apply mel floor
    FBE_infra = H_infra * MAG(1:K,:); % FBE( FBE<1.0 ) = 1.0; % apply mel floor

    % DCT matrix computation
    DCT = dctm( N, M );

    % Conversion of logFBEs to cepstral coefficients through DCT
    CC =  DCT * log( FBE+1e-9 );
    CC_infra =  DCT * log( FBE_infra+1e-9 );

    % Cepstral lifter computation
    lifter = ceplifter( N, L );

    % Cepstral liftering gives liftered cepstral coefficients
    CC = diag( lifter ) * CC; % ~ HTK's MFCCs
    CC_infra = diag( lifter ) * CC_infra; % ~ HTK's MFCCs

    %CC = CC.^2;
end

% EOF
