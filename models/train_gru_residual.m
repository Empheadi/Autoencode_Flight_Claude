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
%                 .maxEpochs       (default 20)
%                 .miniBatchSize   (default 512)
%                 .initialLR       (default 1e-3)
%
%   Outputs:
%     net  — trained dlnetwork
%     info — struct with training history (loss per epoch)

    if nargin < 5, options = struct(); end
    maxEpochs     = getOr(options, 'maxEpochs', 20);
    miniBatchSize = getOr(options, 'miniBatchSize', 512);
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
    lrDropPeriod = 8;
    lrDropFactor = 0.5;

    info.trainLoss = zeros(1, maxEpochs);
    info.valLoss   = zeros(1, maxEpochs);

    % Cap validation samples to avoid OOM
    maxValSamples = min(numVal, 5000);
    valIdx = randperm(numVal, maxValSamples);

    fprintf('  Training: %d samples, %d batches/epoch (batch=%d)\n', ...
        numTrain, ceil(numTrain / miniBatchSize), miniBatchSize);

    for epoch = 1:maxEpochs
        tic;

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

        % Validation loss (on subset, in batches to avoid OOM)
        valLoss = 0;
        valBatches = 0;
        for vb = 1:miniBatchSize:maxValSamples
            vidx = valIdx(vb:min(vb + miniBatchSize - 1, maxValSamples));
            X_v = dlarray(X_val(:, :, vidx), 'CTB');
            Y_v = dlarray(Y_val(:, vidx), 'CB');
            vl = forward(net, X_v);
            valLoss = valLoss + extractdata(mean((vl - Y_v).^2, 'all'));
            valBatches = valBatches + 1;
        end
        info.valLoss(epoch) = valLoss / valBatches;

        elapsed = toc;
        fprintf('Epoch %3d/%d  Train: %.6f  Val: %.6f  LR: %.1e  (%.1fs)\n', ...
            epoch, maxEpochs, info.trainLoss(epoch), info.valLoss(epoch), lr, elapsed);
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
