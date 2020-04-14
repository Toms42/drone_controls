dt = 0.05;
m = 0.458;  % kg
Ix = 4.856*10^-3;  % kg m^2
Iy = 4.856*10^-3;  % kg m^2
Iz = 8.801*10^-3;  % kg m^2
I_vals = [Ix; Iy; Iz];

A = [zeros(3), eye(3), zeros(3,1);
     zeros(3), zeros(3), zeros(3,1);
     zeros(1,3), zeros(1,3), zeros(1,1)];
B = [zeros(3), zeros(3,1);
     eye(3), zeros(3,1);
     zeros(1,3), 1];

g = 9.81;
G = [0 0 0 0 0 -1 0]' * g;
G_ff = [0 0 1 0]' * g;
Q = diag([10,10,10,.1,.1,.1,1]);
R = diag([.1, .1, .1, .1]);

vis = Visualize6DoF(dt);
n = 0;

rp_traj = zeros(10/dt,2);  % [phi, theta]
Kp = [0.1; 0.1; 0.1];  % proportional gains for P controller
Kd = [0.1; 0.1; 0.1];

x = [0 0 0 0 0 0 0]';
rot = eye(3);
x_ref = [3 5 8 0 0 0 pi]';
k = lqr(A, B, Q, R);
w = [0; 0; 0];
for i = 1:dt:10
    tic
    n = n + 1;
    
    % Flat system update
    u = -k*(x - x_ref) + G_ff;
    dx = (A*x + B*u + G);
    target_x = x + dt*dx;
    
    % Dynamic inversion to produce desired attitude and thrust
    up = u(1:3);  % 3 x 1
    upsi = u(4);  % 1 x 1
    thrust = m*sqrt(up'*up);   % Thrust
    psid = target_x(7);
    rot_z = get_rotz(psid);
    z = rot_z * up * (m/-thrust);
    phid = asin(-z(2));
    thetad = atan2(z(1), z(3));
    
    % Calculate angular velocities
    target_rot = eul2rotm([phid; thetad; psid], 'ZYX');
    rot_btwn = rot' * target_rot;
    eul_ang_error = rotm2eul(rot_btwn, 'ZYX');
    eul_ang_vel = eul_ang_error ./ dt;
    
    % Generate new input torques with PD controller
    % https://andrew.gibiansky.com/downloads/pdf/Quadcopter%20Dynamics,%20Simulation,%20and%20Control.pdf
    torques = -I_vals .* (Kd .* eul_ang_vel + Kp .* eul_ang_error);
    
    % Feed current state and inputs to generate updates
    p = x(1:3); v = x(4:6);
    [v_dot, w_dot] = quadcopterDynamics(p, v, rot, w, thrust, torques);
    x(1:3) = p + v*dt;
    x(4:6) = v + v_dot * dt;
    w = w + w_dot * dt;
    rot = applyAngVel(rot, w, dt);

    % Visualize simulated true trajectory
    vis.setRobotState(x(1:3),rot,n);
    vis.setReferenceState(x_ref(1:3),eul2rotm([x_ref(7);0;0],'ZYX'),n);
    pause(min(dt-toc,0));
    vis.showFrame();
end

function[rotz] = get_rotz(yaw)  % psi
    rotz = [cos(yaw), -sin(yaw), 0;
             sin(yaw), cos(yaw), 0;
             0, 0, 1];
end