function M = compute_offline_metrics(omega_dot_true, omega_dot_est, dt)
% COMPUTE_OFFLINE_METRICS  Compute time- and frequency-domain estimation metrics.
%
%   M = compute_offline_metrics(omega_dot_true, omega_dot_est, dt)
%
%   Inputs:
%     omega_dot_true — ground-truth angular acceleration (3×N)
%     omega_dot_est  — estimated angular acceleration (3×N)
%     dt             — sample time (s)
%
%   Output:
%     M — struct with fields:
%           .rmse         (3×1) root-mean-square error per axis
%           .mae          (3×1) mean absolute error per axis
%           .peak_err     (3×1) peak absolute error per axis
%           .delay_samples (3×1) estimated delay in samples
%           .delay_sec    (3×1) estimated delay in seconds
%           .coherence    {3×1 cell} coherence arrays
%           .freq_coh     {3×1 cell} frequency vectors for coherence
%           .gain_dB      {3×1 cell} transfer function magnitude (dB)
%           .phase_deg    {3×1 cell} transfer function phase (deg)
%           .freq_tf      {3×1 cell} frequency vectors for TF

    err = omega_dot_true - omega_dot_est;

    M.rmse     = sqrt(mean(err.^2, 2));          % 3×1
    M.mae      = mean(abs(err), 2);               % 3×1
    M.peak_err = max(abs(err), [], 2);             % 3×1

    Fs = 1 / dt;
    nfft = 256;
    win  = hanning(nfft);
    noverlap = nfft / 2;

    M.delay_samples = zeros(3, 1);
    M.delay_sec     = zeros(3, 1);
    M.coherence     = cell(3, 1);
    M.freq_coh      = cell(3, 1);
    M.gain_dB       = cell(3, 1);
    M.phase_deg     = cell(3, 1);
    M.freq_tf       = cell(3, 1);

    for ax = 1:3
        % Cross-correlation delay
        maxLag = 50;
        [xc, lags] = xcorr(omega_dot_true(ax,:), omega_dot_est(ax,:), maxLag);
        [~, idx] = max(xc);
        M.delay_samples(ax) = lags(idx);
        M.delay_sec(ax)     = M.delay_samples(ax) * dt;

        % Coherence
        [M.coherence{ax}, M.freq_coh{ax}] = mscohere(...
            omega_dot_true(ax,:), omega_dot_est(ax,:), win, noverlap, nfft, Fs);

        % Transfer function estimate
        [Txy, M.freq_tf{ax}] = tfestimate(...
            omega_dot_true(ax,:), omega_dot_est(ax,:), win, noverlap, nfft, Fs);
        M.gain_dB{ax}  = 20*log10(abs(Txy));
        M.phase_deg{ax} = angle(Txy) * 180/pi;
    end
end
