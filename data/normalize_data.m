function [X_norm, norm_stats] = normalize_data(X, norm_stats)
% NORMALIZE_DATA  Channel-wise zero-mean, unit-variance normalization.
%
%   [X_norm, norm_stats] = normalize_data(X)
%   [X_norm, norm_stats] = normalize_data(X, norm_stats)
%
%   Inputs:
%     X          — input array (numChannels × Nwin × numSamples)
%     norm_stats — (optional) struct with .mu and .sigma to reuse
%                  If omitted, stats are computed from X.
%
%   Outputs:
%     X_norm     — normalized array, same size as X
%     norm_stats — struct with fields:
%                    .mu    (numChannels × 1) per-channel mean
%                    .sigma (numChannels × 1) per-channel std

    [numCh, Nwin, numSamples] = size(X);

    if nargin < 2 || isempty(norm_stats)
        % Compute stats: reshape to (numCh, Nwin*numSamples), then stats along dim 2
        X_flat = reshape(X, numCh, []);            % numCh × (Nwin*numSamples)
        norm_stats.mu    = mean(X_flat, 2);        % numCh × 1
        norm_stats.sigma = std(X_flat, 0, 2);      % numCh × 1

        % Avoid division by zero
        norm_stats.sigma(norm_stats.sigma < 1e-10) = 1;
    end

    % Normalize
    X_norm = (X - norm_stats.mu) ./ norm_stats.sigma;
end
