function [dist, dist_state] = disturbance_step(t, P, dist_state)
% DISTURBANCE_STEP  Generate external disturbance torque.
%
%   [dist, dist_state] = disturbance_step(t, P, dist_state)
%
%   Inputs:
%     t          — current time (s)
%     P          — parameter struct (fields: dist_type, dist_std, dist_bw, T, dt)
%     dist_state — struct with field .x (3×1) for band-limited filter state
%
%   Outputs:
%     dist       — disturbance torque (3×1, Nm)
%     dist_state — updated filter state

    switch P.dist_type
        case 'none'
            dist = zeros(3, 1);

        case 'step'
            if t > P.T/3 && t < 2*P.T/3
                dist = P.dist_std;
            else
                dist = zeros(3, 1);
            end

        case 'band_limited'
            w = P.dist_bw * 2 * pi;                 % bandwidth in rad/s
            for i = 1:3
                dist_state.x(i) = dist_state.x(i) ...
                    + P.dt * (-w * dist_state.x(i) + w * P.dist_std(i) * randn);
            end
            dist = dist_state.x;

        case 'sinusoidal'
            dist = P.dist_std .* sin(2*pi*3*t);      % 3 Hz sinusoidal

        otherwise
            error('Unknown dist_type: %s', P.dist_type);
    end
end
