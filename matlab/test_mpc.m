clear;
close all;
%figure;

global T
T = 0.1;

global A B
A = [1 T; 0 1];
B = [T^2/2; T];

global x_bar u_bar
x_bar = 5;
u_bar = 1;

L_x = [0  1;
       1  0;
       0 -1;
      -1  0];
b_x = [x_bar; x_bar; x_bar; x_bar];

L_u = [1; -1];
b_u = [u_bar; u_bar];

global N
N = 1/T;

Q = eye(2);
S = 10*eye(2);
R = eye(1);

Q_bar = bigblockdiag(Q,N+1);
Q_bar(end-1:end,end-1:end) = S;
R_bar = bigblockdiag(R,N);
A_stack = build_a_bar(A,N+1);
B_bar = build_b_bar(A,B,N+1);
L_u_bar = bigblockdiag(L_u,N);
L_x_bar = bigblockdiag(L_x,N+1);
b_u_bar = bigstack(b_u, N);
b_x_bar = bigstack(b_x, N+1);

L = [L_u_bar; L_x_bar * B_bar];

global x0
x0 = [2;0];
x_k = x0;


iter = 150;
final_trajectory = zeros((iter)*2,1);
X_trajectories = cell(iter+1);
final_u = zeros(iter,1);
ref = generate_reference_traj(iter);
tic
for k = 0:iter-1
    f_bar = A_stack * x_k;
    G_bar = ref(2*k+1:2*(k+N+1));
    
    % Compute quadprog matrices:
    H = B_bar' * Q_bar * B_bar + R_bar;
    C = f_bar' * Q_bar * B_bar - G_bar' * Q_bar * B_bar;
    b = [b_u_bar; b_x_bar - L_x_bar * f_bar];
    evalc('U = quadprog(H, C, L, b)');
    
    X_new = f_bar + B_bar * U;
    final_u(k+1) = U(1);
    X_trajectories{k+1} = reshape(X_new,2,N+1);
    x_k = X_new(3:4);
    final_trajectory(k*2+1:k*2+2) = x_k;
end
toc

final_trajectory = reshape(final_trajectory,2,iter);

cmap = colormap('jet');
colors = [resample(cmap(:,1), iter, size(cmap, 1))'; ...
    resample(cmap(:,2), iter, size(cmap, 1))'; ...
    resample(cmap(:,3), iter, size(cmap, 1))'];
colors = max(min(colors,1),0);

% Spatial plot
reference_trajectory = reshape(generate_reference_traj(iter),2,iter + N);
plot(reference_trajectory(1,:), reference_trajectory(2,:), 'color', [1;0;0])
hold on
for i = 1:iter
    plot(X_trajectories{i}(1,:),X_trajectories{i}(2,:), 'color', colors(:,i));
end
plot(final_trajectory(1,:), final_trajectory(2,:), 'color', [0;0;0]);
xlabel('x1')
ylabel('x2')
title('MPC trajectory for 150 iterations with N=10s');

figure;
% Time plot
t = T:T:(N+iter)*T;
plot(t, reference_trajectory(1,:), 'color', [1;0;0]);
hold on
plot(t, reference_trajectory(2,:), 'color', [1;0;0]);
for i = 1:iter
    plot([i/10-.1:.1:i/10+N/10-0.1], X_trajectories{i}(1,:), 'color', colors(:,i));
    plot([i/10-.1:.1:i/10+N/10-0.1], X_trajectories{i}(2,:), 'color', colors(:,i));
end
t = T:T:iter*T;
plot(t, final_trajectory(1,:), 'color', [0;0;0]);
plot(t, final_trajectory(2,:), 'color', [0;0;0]);
xlabel('t')
title('MPC trajectory for 150 iterations with N=10s');

figure;
plot(1:iter,final_u);

function G = generate_reference_traj(iter)
    global x0 T N
    G = zeros(2*(iter + N), 1);
    for t = T:T:10
        G(round(2*t/T-1):round(2*t/T)) = (10 - t)/(10) * x0;
    end
end

function blk = bigblockdiag(M, n)
    blk = zeros(size(M)*n);
    
    rm = size(M,1);
    cm = size(M,2);
    
    for i = 1:n
        blk(rm*(i-1)+1:rm*i, cm*(i-1)+1:cm*i) = M;
    end
end

function stk = bigstack(M,n)
    rm = size(M,1);
    cm = size(M,2);
    stk = zeros(rm*n,cm);
    
    for i = 1:n
        stk(rm*(i-1)+1:rm*i,:) = M;
    end
end

function A_bar = build_a_bar(A, n)
    rm = size(A,1);
    cm = size(A,2);
    A_bar = zeros(rm*n,cm);
    
    for i = 1:n
        A_bar(rm*(i-1)+1:rm*i,:) = A^(i-1);
    end
end

function B_bar = build_b_bar(A, B, n)
    rm = size(B,1);
    cm = size(B,2);
    B_bar = zeros(rm*n,cm*(n-1));
    for r = 1:n
        for c = 1:n-1
            order = r-c-1;
            if order < 0
                B_bar(rm*(r-1)+1:rm*r, cm*(c-1)+1:cm*c) = zeros(size(B));
            else
                B_bar(rm*(r-1)+1:rm*r, cm*(c-1)+1:cm*c) = A^order * B;
            end
        end
    end
end