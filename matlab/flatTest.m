close all
dt = 0.05;

A = [zeros(3), eye(3), zeros(3,1);
     zeros(3), zeros(3), zeros(3,1);
     zeros(1,3), zeros(1,3), zeros(1,1)];
B = [zeros(3), zeros(3,1);
     eye(3), zeros(3,1);
     zeros(1,3), 1];

g = 9.81;
G = [0 0 0 0 0 -1 0]' * g;
%G_ff = [0 0 1 0]' * g;
Q = diag([10,10,10,.1,.1,.1,1]);
R = diag([.1, .1, .1, .1]);


waypts = [0 0 0; 1 2 3; 4 2 3; 4 4 4]';
v0 = zeros(3,1);
a0 = zeros(3,1);
v1 = zeros(3,1);
z1 = zeros(3,1);
[xx, yy, zz, vxx, vyy, vzz, axx, ayy, azz] = constructMinimumSnapTraj(dt,3,waypts,v0,a0,v1,a1);
psi = atan2(vyy,vxx);
psi(end) = psi(end-1);
ref_traj = [xx;yy;zz;vxx;vyy;vzz;psi];
feedforward = [axx; ayy; azz + ones(size(axx)) * g; zeros(size(axx))];

vis = Visualize6DoF(dt);
vis.setReferenceTraj(ref_traj);

x = [0 0 0 0 0 0 0]';
k = lqr(A, B, Q, R);
for n = 1:size(ref_traj,2)+1/dt
    tic
    
    x_ref = ref_traj(:,min(n,size(ref_traj,2)));
    if n > size(feedforward,2)
        ff = zeros(4,1);
    else
        ff = feedforward(:,n);
    end
    
    u = -k*(x - x_ref) + ff;
    x = x + dt*(A*x + B*u + G);
    
    vis.setRobotState(x(1:3),eul2rotm([x(7),0,0],'zyx'),n);
    vis.setReferenceState(x_ref(1:3),eul2rotm([x_ref(7),0,0],'zyx'),n);
    pause(min(dt-toc,0));
    vis.showFrame();
end