% MAIN_MONTECARLO  Monte Carlo robustness study across estimators.
%
%   Runs N_mc simulations with randomized parameters for each estimator
%   and produces comparison plots.

rng(42);
addpath('sim', 'estimators', 'utils', 'eval', 'models');

P = init_params();
N_mc = 200;

estimators = {@est_backdiff_lpf_step, @est_eso_step, @est_compfilt_step};
est_names  = {'BackDiff', 'ESO', 'CompFilt'};
if exist('models/gru_residual_trained.mat', 'file')
    estimators{end+1} = @est_learned_step;
    est_names{end+1}  = 'GRU-Res';
end

numEst = length(estimators);

% Standard scenario: roll step
omega_cmd_fun = @(t) [deg2rad(25)*(t>1 & t<3); 0; 0];

% Shorten episode for MC speed
P_mc_base = P;
P_mc_base.T = 10;

% Pre-allocate results
results = struct('tracking_rmse', cell(N_mc, numEst), ...
                 'peak_error',    cell(N_mc, numEst), ...
                 'actuator_rms',  cell(N_mc, numEst));

fprintf('Running %d Monte Carlo iterations with %d estimators...\n', N_mc, numEst);
for i = 1:N_mc
    rng(5000 + i);
    P_mc = randomize_params(P_mc_base);

    for j = 1:numEst
        rng(5000 + i);  % same noise realization per estimator
        L = run_episode(P_mc, estimators{j}, omega_cmd_fun);
        M = compute_closedloop_metrics(L, P_mc);
        results(i, j).tracking_rmse = M.tracking_rmse;
        results(i, j).peak_error    = M.peak_error;
        results(i, j).actuator_rms  = M.actuator_rms;
    end

    if mod(i, 50) == 0
        fprintf('  MC iteration %d / %d\n', i, N_mc);
    end
end

% ---- Summary statistics ----
fprintf('\n=== Monte Carlo Summary (Roll Axis, %d runs) ===\n', N_mc);
fprintf('%-10s | RMSE mean±std (deg/s) | Peak mean±std (deg/s)\n', 'Estimator');
fprintf('%s\n', repmat('-', 1, 60));
for j = 1:numEst
    rmse_vals = zeros(N_mc, 1);
    peak_vals = zeros(N_mc, 1);
    for i = 1:N_mc
        rmse_vals(i) = rad2deg(results(i, j).tracking_rmse(1));
        peak_vals(i) = rad2deg(results(i, j).peak_error(1));
    end
    fprintf('%-10s | %6.3f ± %5.3f        | %6.2f ± %5.2f\n', ...
        est_names{j}, mean(rmse_vals), std(rmse_vals), mean(peak_vals), std(peak_vals));
end

% ---- Plots ----
plot_mc_results(results, est_names);

% Save results
save('eval/mc_results.mat', 'results', 'est_names', 'N_mc', '-v7.3');
fprintf('\nMonte Carlo study complete. Results saved to eval/mc_results.mat\n');
