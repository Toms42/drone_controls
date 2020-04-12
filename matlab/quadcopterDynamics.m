function [v_dot, w_dot] = quadcopterDynamics(p, v, R, w, c, tau)
%QUADCOPTERTDYNAMICS Summary of this function goes here
%   Detailed explanation goes here

I = diag([1 1 1]);
A = zeros(3);
B = zeros(3);
D = zeros(3);
g = 9.81;

z_w = [0 0 1]';
z_b = R(1:3, 3);
tau_g = [0 0 0]';

v_dot = -g * z_w + c*z_b - R*D*R'*v;
w_dot = inv(I) * (tau - cross(w, I*w) - tau_g - A*R'*v - B*w);
end