function plot_mc_results(results, est_names)
% PLOT_MC_RESULTS  Plot Monte Carlo robustness study results.
%
%   plot_mc_results(results, est_names)
%
%   Inputs:
%     results   — struct array (N_mc × num_estimators) with fields:
%                    .tracking_rmse (3×1)
%                    .peak_error    (3×1)
%                    .actuator_rms  (3×1)
%     est_names — cell array of estimator name strings

    [N_mc, numEst] = size(results);

    % Extract roll-axis tracking RMSE for simplicity
    rmse_data = zeros(N_mc, numEst);
    for j = 1:numEst
        for i = 1:N_mc
            rmse_data(i, j) = rad2deg(results(i, j).tracking_rmse(1));  % roll axis, deg/s
        end
    end

    % ---- 1. CDF of tracking RMSE ----
    figure('Name', 'MC CDF', 'Position', [50 50 700 400]);
    colors = lines(numEst);
    for j = 1:numEst
        sorted = sort(rmse_data(:, j));
        cdf_y  = (1:N_mc)' / N_mc;
        plot(sorted, cdf_y, 'Color', colors(j,:), 'LineWidth', 1.3, ...
             'DisplayName', est_names{j}); hold on;
    end
    xlabel('Roll Tracking RMSE (deg/s)');
    ylabel('CDF');
    title('CDF of Tracking RMSE (Roll Axis)');
    legend('Location', 'best');
    grid on;

    % ---- 2. Box plot of RMSE ----
    figure('Name', 'MC Box', 'Position', [50 50 600 400]);
    boxplot(rmse_data, 'Labels', est_names);
    ylabel('Roll Tracking RMSE (deg/s)');
    title('MC Distribution of Tracking RMSE');
    grid on;

    % ---- 3. Multi-axis RMSE bar chart (mean ± std) ----
    axis_names = {'Roll', 'Pitch', 'Yaw'};
    figure('Name', 'MC Bar', 'Position', [50 50 800 400]);
    for ax = 1:3
        data_ax = zeros(N_mc, numEst);
        for j = 1:numEst
            for i = 1:N_mc
                data_ax(i, j) = rad2deg(results(i, j).tracking_rmse(ax));
            end
        end
        subplot(1, 3, ax);
        bar_means = mean(data_ax, 1);
        bar_stds  = std(data_ax, 0, 1);
        bar(bar_means); hold on;
        errorbar(1:numEst, bar_means, bar_stds, 'k.', 'LineWidth', 1.2);
        set(gca, 'XTickLabel', est_names, 'XTickLabelRotation', 30);
        ylabel('RMSE (deg/s)');
        title(axis_names{ax});
        grid on;
    end
end
