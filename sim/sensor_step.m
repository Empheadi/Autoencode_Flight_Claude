function omega_meas = sensor_step(omega_true, bias, P)
% SENSOR_STEP  Simulate noisy gyroscope measurement.
%
%   omega_meas = sensor_step(omega_true, bias, P)
%
%   Inputs:
%     omega_true — true angular rate (3×1, rad/s)
%     bias       — gyro bias (3×1, rad/s), drawn once per episode
%     P          — parameter struct (fields: gyro_noise_std, gyro_quantize)
%
%   Output:
%     omega_meas — measured angular rate (3×1, rad/s)

    noise = P.gyro_noise_std .* randn(3, 1);          % 3×1
    omega_meas = omega_true + bias + noise;

    if P.gyro_quantize > 0
        omega_meas = round(omega_meas / P.gyro_quantize) * P.gyro_quantize;
    end
end
