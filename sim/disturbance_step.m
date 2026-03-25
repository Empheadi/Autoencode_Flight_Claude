function [dist, dist_accel, dist_state] = disturbance_step(t, P, dist_state)
% DISTURBANCE_STEP  Generate external disturbance torque and acceleration.
%
%   [dist, dist_accel, dist_state] = disturbance_step(t, P, dist_state)
%
%   Inputs:
%     t          — current time (s)
%     P          — parameter struct (fields: dist_type, dist_std, dist_bw,
%                  dist_accel_std, T, dt)
%     dist_state — struct with fields .x (3×1) and .xa (3×1) filter states
%
%   Outputs:
%     dist       — disturbance torque (3×1, Nm)
%     dist_accel — acceleration-level disturbance (3×1, rad/s^2), bypasses I
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
            w = P.dist_bw * 2 * pi;
            for i = 1:3
                dist_state.x(i) = dist_state.x(i) ...
                    + P.dt * (-w * dist_state.x(i) + w * P.dist_std(i) * randn);
            end
            dist = dist_state.x;

        case 'sinusoidal'
            dist = P.dist_std .* sin(2*pi*3*t);

        case 'ramp'
            ramp = min(t / (P.T * 0.8), 1.0);
            dist = P.dist_std * ramp;

        otherwise
            error('Unknown dist_type: %s', P.dist_type);
    end

    % ---- Acceleration-level disturbance (bypasses inertia) ----
    if isfield(P, 'dist_accel_std') && any(P.dist_accel_std ~= 0)
        if ~isfield(dist_state, 'xa')
            dist_state.xa = zeros(3, 1);
        end
        w = P.dist_bw * 2 * pi;
        for i = 1:3
            dist_state.xa(i) = dist_state.xa(i) ...
                + P.dt * (-w * dist_state.xa(i) + w * P.dist_accel_std(i) * randn);
        end
        dist_accel = dist_state.xa;
    else
        dist_accel = zeros(3, 1);
    end
end
