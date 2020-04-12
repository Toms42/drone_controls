dt = 0.05;

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

x = [0 0 0 0 0 0 0]';
x_ref = [3 5 8 0 0 0 pi]';
k = lqr(A, B, Q, R);
for i = 1:dt:10
    tic
    n = n + 1;
    
    u = -k*(x - x_ref) + G_ff;
    x = x + dt*(A*x + B*u + G);
    
    vis.setRobotState(x(1:3),eul2rotm([x(7),0,0],'zyx'),n);
    vis.setReferenceState(x_ref(1:3),eul2rotm([x_ref(7),0,0],'zyx'),n);
    pause(min(dt-toc,0));
    vis.showFrame();
end