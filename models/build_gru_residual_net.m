function net = build_gru_residual_net(numChannels, Nwin)
% BUILD_GRU_RESIDUAL_NET  Create a two-layer GRU network for residual prediction.
%
%   net = build_gru_residual_net(numChannels, Nwin)
%
%   Inputs:
%     numChannels — number of input channels (default 10)
%     Nwin        — sequence length (used for documentation; not hard-coded in layers)
%
%   Output:
%     net — dlnetwork ready for training
%           Input:  numChannels × Nwin sequence
%           Output: 3×1 residual angular acceleration

    if nargin < 1, numChannels = 10; end
    if nargin < 2, Nwin = 25; end  %#ok<NASGU> — for reference only

    layers = [
        sequenceInputLayer(numChannels, 'Name', 'input')
        gruLayer(64, 'OutputMode', 'last', 'Name', 'gru1')
        gruLayer(32, 'OutputMode', 'last', 'Name', 'gru2')
        fullyConnectedLayer(3, 'Name', 'fc_out')
    ];

    net = dlnetwork(layers);
end
