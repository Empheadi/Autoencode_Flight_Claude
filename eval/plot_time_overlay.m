function plot_time_overlay(L, estimates, est_names, t_range)
% PLOT_TIME_OVERLAY  Plot true vs estimated angular acceleration for multiple estimators.
%
%   plot_time_overlay(L, estimates, est_names)
%   plot_time_overlay(L, estimates, est_names, t_range)
%
%   Inputs:
%     L          — log struct from run_episode (must have omega_dot_true, t)
%     estimates  — cell array of 3×N matrices (omega_dot_est for each estimator)
%     est_names  — cell array of strings (estimator names)
%     t_range    — (optional) [t_start t_end] in seconds for zoom

    axis_names = {'Roll (p)', 'Pitch (q)', 'Yaw (r)'};
    colors = lines(length(estimates));

    if nargin < 4
        t_range = [L.t(1), L.t(end)];
    end

    idx = L.t >= t_range(1) & L.t <= t_range(2);

    figure('Name', 'Acceleration Overlay', 'Position', [50 50 1000 700]);
    for ax = 1:3
        subplot(3, 1, ax);
        plot(L.t(idx), rad2deg(L.omega_dot_true(ax, idx)), 'k', ...
             'LineWidth', 1.3, 'DisplayName', 'Truth'); hold on;

        for j = 1:length(estimates)
            plot(L.t(idx), rad2deg(estimates{j}(ax, idx)), ...
                 'Color', colors(j,:), 'LineWidth', 0.9, ...
                 'DisplayName', est_names{j});
        end

        ylabel([axis_names{ax}, ' (deg/s^2)']);
        legend('Location', 'best');
        grid on;
        if ax == 1, title('Angular Acceleration: Truth vs Estimators'); end
    end
    xlabel('Time (s)');
end
