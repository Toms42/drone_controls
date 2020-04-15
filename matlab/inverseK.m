function [phi, theta, psi, thrust] = inverseK(x, u, m)
%INVERSEK Summary of this function goes here
%   Detailed explanation goes here
    up = [u(1), u(2), u(3)]';
    thrust = m*sqrt(up'*up);   % Thrust
    psid = x(7);
    rot_z = eul2rotm([-psid 0 0], 'ZYX');
    z = rot_z * up / (norm(up));
    phid = -atan2(z(2), z(3));
    thetad = atan2(z(1), z(3));
    
    phi = phid;
    theta = thetad;
    psi = psid;
end

