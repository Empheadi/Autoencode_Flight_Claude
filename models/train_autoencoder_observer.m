function [net, info] = train_autoencoder_observer(X_train, Y_omega_train, Y_res_train, ...
                            omega_dot_base_train, X_val, Y_omega_val, Y_res_val, ...
                            omega_dot_base_val, options)
% TRAIN_AUTOENCODER_OBSERVER  Train the two-head autoencoder observer network.
%
%   Inputs:
%     X_train              — training inputs (numCh × Nwin × numSamples)
%     Y_omega_train        — true omega at prediction index (3 × numSamples)
%     Y_res_train          — residual target (3 × numSamples)
%     omega_dot_base_train — base estimator output (3 × numSamples)
%     X_val, Y_omega_val, Y_res_val, omega_dot_base_val — validation data
%     options              — struct with .maxEpochs, .miniBatchSize, .initialLR,
%                            .lambda (2×1: [accel_weight, rate_weight])
%
%   Outputs:
%     net  — trained dlnetwork
%     info — training history

    if nargin < 9, options = struct(); end
    maxEpochs     = getOr(options, 'maxEpochs', 20);
    miniBatchSize = getOr(options, 'miniBatchSize', 512);
    initialLR     = getOr(options, 'initialLR', 1e-3);
    lambda        = getOr(options, 'lambda', [1.0, 0.3]);

    [numCh, Nwin, numTrain] = size(X_train);
    numVal = size(X_val, 3);

    % Build network
    net = build_autoencoder_observer_net(numCh, Nwin);

    % Adam state
    avgGrad   = [];
    avgSqGrad = [];
    iteration = 0;

    lrDropPeriod = 8;
    lrDropFactor = 0.5;

    info.trainLoss = zeros(1, maxEpochs);
    info.valLoss   = zeros(1, maxEpochs);

    % Cap validation samples
    maxValSamples = min(numVal, 5000);
    valIdx = randperm(numVal, maxValSamples);

    fprintf('  Training: %d samples, %d batches/epoch (batch=%d)\n', ...
        numTrain, ceil(numTrain / miniBatchSize), miniBatchSize);

    for epoch = 1:maxEpochs
        tic;
        perm = randperm(numTrain);
        epochLoss = 0;
        numBatches = 0;
        lr = initialLR * lrDropFactor^(floor((epoch-1) / lrDropPeriod));

        for b = 1:miniBatchSize:numTrain
            iteration = iteration + 1;
            idx = perm(b:min(b + miniBatchSize - 1, numTrain));

            X_b     = dlarray(X_train(:, :, idx), 'CTB');
            Yom_b   = dlarray(Y_omega_train(:, idx), 'CB');
            Yres_b  = dlarray(Y_res_train(:, idx), 'CB');
            base_b  = dlarray(omega_dot_base_train(:, idx), 'CB');

            [loss, grads] = dlfeval(@modelLoss, net, X_b, Yom_b, Yres_b, base_b, lambda);

            [net, avgGrad, avgSqGrad] = adamupdate(net, grads, ...
                avgGrad, avgSqGrad, iteration, lr);

            epochLoss = epochLoss + extractdata(loss);
            numBatches = numBatches + 1;
        end

        info.trainLoss(epoch) = epochLoss / numBatches;

        % Validation (batched, on subset)
        valLoss = 0;
        valBatches = 0;
        for vb = 1:miniBatchSize:maxValSamples
            vi = valIdx(vb:min(vb + miniBatchSize - 1, maxValSamples));
            X_v    = dlarray(X_val(:, :, vi), 'CTB');
            Yom_v  = dlarray(Y_omega_val(:, vi), 'CB');
            Yres_v = dlarray(Y_res_val(:, vi), 'CB');
            base_v = dlarray(omega_dot_base_val(:, vi), 'CB');

            [oh, dod] = forward(net, X_v, 'Outputs', {'h1_out', 'h2_out'});
            L1 = mean((base_v + dod - base_v - Yres_v).^2, 'all');
            L2 = mean((oh - Yom_v).^2, 'all');
            valLoss = valLoss + extractdata(lambda(1)*L1 + lambda(2)*L2);
            valBatches = valBatches + 1;
        end
        info.valLoss(epoch) = valLoss / valBatches;

        elapsed = toc;
        fprintf('Epoch %3d/%d  Train: %.6f  Val: %.6f  LR: %.1e  (%.1fs)\n', ...
            epoch, maxEpochs, info.trainLoss(epoch), info.valLoss(epoch), lr, elapsed);
    end
end

function [loss, grads] = modelLoss(net, X, Y_omega, Y_res, base, lam)
    [omega_hat, delta_omegadot] = forward(net, X, 'Outputs', {'h1_out', 'h2_out'});
    omegadot_target = base + Y_res;
    omegadot_hat    = base + delta_omegadot;
    L1 = mean((omegadot_hat - omegadot_target).^2, 'all');
    L2 = mean((omega_hat - Y_omega).^2, 'all');
    loss = lam(1)*L1 + lam(2)*L2;
    grads = dlgradient(loss, net.Learnables);
end

function v = getOr(s, field, default)
    if isfield(s, field), v = s.(field); else, v = default; end
end
