R = eye(3);
p = [0;0;0];
v = [0;0;0];
w = [0;0;0];

freq = 10;
dt = 1/freq;
vis = Visualize6DoF(dt);
joy = vrjoystick(1);
n = 0;
while true
    n = n + 1;
    tic;
    [c, tau] = joy2ct(joy);
    [v_dot, w_dot] = quadcopterDynamics(p, v, R, w, c, tau);
    p = p + v*dt;
    v = v + v_dot * dt;
    w = w + w_dot * dt;
    R = applyAngVel(R, w, dt);
    vis.setRobotState(p,R,n);
    vis.setReferenceState([0;0;0],eye(3),n);
    vis.showFrame();
    pause(min(dt-toc,0));
end


function [c,tau] = joy2ct(joy)
    idleThrust = 0;%9.81;
    tau = zeros(3,1);
    
    c = -axis(joy, 2)                     * 40;
    tau(3) = -axis(joy, 1)                * 1;
    tau(1) = axis(joy,4)                  * 1;
    tau(2) = -axis(joy,5)                 * 1;
    
    tau = tau .* double(abs(tau) > 0.1);
    c = c .* double(abs(c) > 0.1);
    c = c + idleThrust;
end