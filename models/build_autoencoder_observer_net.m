function net = build_autoencoder_observer_net(numChannels, Nwin)
% BUILD_AUTOENCODER_OBSERVER_NET  Create a two-head GRU network.
%
%   net = build_autoencoder_observer_net(numChannels, Nwin)
%
%   Architecture:
%     Shared encoder:  GRU(64) → GRU(32) → latent (32×1)
%     Head 1 (rate):   FC(16) → ReLU → FC(3)  → omega_hat (denoised rate)
%     Head 2 (accel):  FC(16) → ReLU → FC(3)  → Delta_omega_dot (residual)
%
%   Inputs:
%     numChannels — number of input channels (default 10)
%     Nwin        — sequence length (default 25, for reference)
%
%   Output:
%     net — dlnetwork with two output branches

    if nargin < 1, numChannels = 10; end
    if nargin < 2, Nwin = 25; end  %#ok<NASGU>

    lgraph = layerGraph();

    % ---- Shared encoder ----
    lgraph = addLayers(lgraph, [
        sequenceInputLayer(numChannels, 'Name', 'input')
        gruLayer(64, 'OutputMode', 'last', 'Name', 'gru1')
        gruLayer(32, 'OutputMode', 'last', 'Name', 'gru2')
    ]);

    % ---- Head 1: denoised rate ----
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(16, 'Name', 'h1_fc1')
        reluLayer('Name', 'h1_relu')
        fullyConnectedLayer(3, 'Name', 'h1_out')
    ]);
    lgraph = connectLayers(lgraph, 'gru2', 'h1_fc1');

    % ---- Head 2: acceleration residual ----
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(16, 'Name', 'h2_fc1')
        reluLayer('Name', 'h2_relu')
        fullyConnectedLayer(3, 'Name', 'h2_out')
    ]);
    lgraph = connectLayers(lgraph, 'gru2', 'h2_fc1');

    % Build dlnetwork
    net = dlnetwork(lgraph);
end
