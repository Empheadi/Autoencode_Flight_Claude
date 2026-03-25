function [net, info] = train_gru_residual(X_train, Y_train, X_val, Y_val, options)
% TRAIN_GRU_RESIDUAL  Train the GRU residual network.
%
%   [net, info] = train_gru_residual(X_train, Y_train, X_val, Y_val, options)
%
%   Inputs:
%     X_train — training inputs (numChannels × Nwin × numSamples)
%     Y_train — training targets (3 × numSamples), residual
%     X_val   — validation inputs
%     Y_val   — validation targets
%     options — (optional) struct with fields:
%                 .maxEpochs       (default 50)
%                 .miniBatchSize   (default 128)
%                 .initialLR       (default 1e-3)
%
%   Outputs:
%     net  — trained dlnetwork
%     info — struct with training history (loss per epoch)

    if nargin < 5, options = struct(); end
    maxEpochs     = getOr(options, 'maxEpochs', 50);
    miniBatchSize = getOr(options, 'miniBatchSize', 128);
    initialLR     = getOr(options, 'initialLR', 1e-3);

    [numCh, Nwin, numTrain] = size(X_train);
    numVal = size(X_val, 3);

    % Build network
    net = build_gru_residual_net(numCh, Nwin);

    % Adam state
    avgGrad   = [];
    avgSqGrad = [];
    iteration = 0;

    % Learning rate schedule
    lrDropPeriod = 15;
    lrDropFactor = 0.5;

    info.trainLoss = zeros(1, maxEpochs);
    info.valLoss   = zeros(1, maxEpochs);

    for epoch = 1:maxEpochs
        % Shuffle training data
        perm = randperm(numTrain);

        epochLoss = 0;
        numBatches = 0;

        % Learning rate
        lr = initialLR * lrDropFactor^(floor((epoch-1) / lrDropPeriod));

        for b = 1:miniBatchSize:numTrain
            iteration = iteration + 1;
            idx = perm(b:min(b + miniBatchSize - 1, numTrain));

            % Prepare mini-batch
            X_batch = dlarray(X_train(:, :, idx), 'CTB');   % Channel × Time × Batch
            Y_batch = dlarray(Y_train(:, idx), 'CB');       % Channel × Batch

            % Compute loss and gradients
            [loss, grads] = dlfeval(@modelLoss, net, X_batch, Y_batch);

            % Adam update
            [net, avgGrad, avgSqGrad] = adamupdate(net, grads, ...
                avgGrad, avgSqGrad, iteration, lr);

            epochLoss = epochLoss + extractdata(loss);
            numBatches = numBatches + 1;
        end

        info.trainLoss(epoch) = epochLoss / numBatches;

        % Validation loss
        X_v = dlarray(X_val, 'CTB');
        Y_v = dlarray(Y_val, 'CB');
        valLoss = dlfeval(@modelLoss, net, X_v, Y_v);
        info.valLoss(epoch) = extractdata(valLoss);

        if mod(epoch, 5) == 0 || epoch == 1
            fprintf('Epoch %3d/%d  Train MSE: %.6f  Val MSE: %.6f  LR: %.1e\n', ...
                epoch, maxEpochs, info.trainLoss(epoch), info.valLoss(epoch), lr);
        end
    end
end

function [loss, grads] = modelLoss(net, X, Y)
    Y_pred = forward(net, X);           % 3 × Batch
    loss = mean((Y_pred - Y).^2, 'all');
    grads = dlgradient(loss, net.Learnables);
end

function v = getOr(s, field, default)
    if isfield(s, field), v = s.(field); else, v = default; end
end
