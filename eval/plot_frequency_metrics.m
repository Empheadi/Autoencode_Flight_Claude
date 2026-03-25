function plot_frequency_metrics(metrics_cell, est_names)
% PLOT_FREQUENCY_METRICS  Plot Bode-like and coherence plots for multiple estimators.
%
%   plot_frequency_metrics(metrics_cell, est_names)
%
%   Inputs:
%     metrics_cell — cell array of metric structs from compute_offline_metrics
%     est_names    — cell array of estimator name strings

    axis_names = {'Roll', 'Pitch', 'Yaw'};
    colors = lines(length(metrics_cell));

    % ---- Bode-like: magnitude + phase ----
    figure('Name', 'Bode Plot', 'Position', [50 50 1100 800]);
    for ax = 1:3
        % Magnitude
        subplot(3, 2, 2*(ax-1)+1);
        for j = 1:length(metrics_cell)
            semilogx(metrics_cell{j}.freq_tf{ax}, metrics_cell{j}.gain_dB{ax}, ...
                     'Color', colors(j,:), 'LineWidth', 1.0, ...
                     'DisplayName', est_names{j}); hold on;
        end
        ylabel('Magnitude (dB)');
        if ax == 1, title('Transfer Function Magnitude'); end
        legend('Location', 'best');
        grid on;
        xlim([1, 200]);

        % Phase
        subplot(3, 2, 2*(ax-1)+2);
        for j = 1:length(metrics_cell)
            semilogx(metrics_cell{j}.freq_tf{ax}, metrics_cell{j}.phase_deg{ax}, ...
                     'Color', colors(j,:), 'LineWidth', 1.0, ...
                     'DisplayName', est_names{j}); hold on;
        end
        ylabel('Phase (deg)');
        if ax == 1, title('Transfer Function Phase'); end
        grid on;
        xlim([1, 200]);
    end
    subplot(3, 2, 5); xlabel('Frequency (Hz)');
    subplot(3, 2, 6); xlabel('Frequency (Hz)');

    % ---- Coherence ----
    figure('Name', 'Coherence', 'Position', [50 50 900 600]);
    for ax = 1:3
        subplot(3, 1, ax);
        for j = 1:length(metrics_cell)
            semilogx(metrics_cell{j}.freq_coh{ax}, metrics_cell{j}.coherence{ax}, ...
                     'Color', colors(j,:), 'LineWidth', 1.0, ...
                     'DisplayName', est_names{j}); hold on;
        end
        ylabel(['Coh — ', axis_names{ax}]);
        ylim([0, 1.05]);
        legend('Location', 'best');
        grid on;
        if ax == 1, title('Coherence vs Frequency'); end
    end
    xlabel('Frequency (Hz)');
end
