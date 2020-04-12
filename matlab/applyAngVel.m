function [R] = applyAngVel(R_0, w, dt)
%APPLYANGVEL Summary of this function goes here
%   Detailed explanation goes here
eps = 1e-9;
if norm(w) < eps
    R_inc = eye(3);
else
    R_inc = axang2rotm([w/norm(w); norm(w)*dt]');
end
R = R_0 * R_inc;
end

