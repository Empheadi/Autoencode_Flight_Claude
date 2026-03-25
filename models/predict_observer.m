function [omega_hat, delta_omegadot] = predict_observer(net, X_norm)
% PREDICT_OBSERVER  Run the autoencoder observer network in inference mode.
%
%   [omega_hat, delta_omegadot] = predict_observer(net, X_norm)
%
%   Inputs:
%     net    — trained dlnetwork (autoencoder observer)
%     X_norm — normalized input (numChannels × Nwin), single sample
%
%   Outputs:
%     omega_hat       — denoised rate estimate (3×1)
%     delta_omegadot  — acceleration residual (3×1)

    X_dl = dlarray(X_norm, 'CT');
    [omega_hat_dl, delta_dl] = predict(net, X_dl, 'Outputs', {'h1_out', 'h2_out'});
    omega_hat      = extractdata(omega_hat_dl);
    delta_omegadot = extractdata(delta_dl);
end
