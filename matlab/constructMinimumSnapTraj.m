function [xx,yy,zz,vxx,vyy,vzz,axx,ayy,azz,jxx,jyy,jzz] = constructMinimumSnapTraj(dt,T,waypts,v0,a0,v1,a1)
%CONSTRUCTMINIMUMSNAPTRAJ Summary of this function goes here
%   Detailed explanation goes herefunction demo1_minimum_snap_simple()
% maoshuyuan123@gmail.com

    % condition
    ts = arrangeT(waypts,T);
    n_order = 5;
    
    % trajectory plan
    polys_x = minimum_snap_single_axis_simple(waypts(1,:),ts,n_order,v0(1),a0(1),v1(1),a1(1));
    polys_y = minimum_snap_single_axis_simple(waypts(2,:),ts,n_order,v0(2),a0(2),v1(2),a1(2));
    polys_z = minimum_snap_single_axis_simple(waypts(3,:),ts,n_order,v0(3),a0(3),v1(3),a1(3));
    
    % result show
    tt = ts(1):dt:ts(end);
    xx = polys_vals(polys_x,ts,tt,0);
    yy = polys_vals(polys_y,ts,tt,0);
    zz = polys_vals(polys_z,ts,tt,0);

    vxx = polys_vals(polys_x,ts,tt,1);
    axx = polys_vals(polys_x,ts,tt,2);
    jxx = polys_vals(polys_x,ts,tt,3);
    vyy = polys_vals(polys_y,ts,tt,1);
    ayy = polys_vals(polys_y,ts,tt,2);
    jyy = polys_vals(polys_y,ts,tt,3);
    vzz = polys_vals(polys_z,ts,tt,1);
    azz = polys_vals(polys_z,ts,tt,2);
    jzz = polys_vals(polys_z,ts,tt,3);
    
    subplot(4,3,1),plot(tt,xx);title('x position');
    subplot(4,3,2),plot(tt,yy);title('y position');
    subplot(4,3,3),plot(tt,zz);title('y position');
    subplot(4,3,4),plot(tt,vxx);title('x velocity');
    subplot(4,3,5),plot(tt,vyy);title('y velocity');
    subplot(4,3,6),plot(tt,vzz);title('y velocity');
    subplot(4,3,7),plot(tt,axx);title('x acceleration');
    subplot(4,3,8),plot(tt,ayy);title('y acceleration');
    subplot(4,3,9),plot(tt,azz);title('y acceleration');
    subplot(4,3,10),plot(tt,jxx);title('x jerk');
    subplot(4,3,11),plot(tt,jyy);title('y jerk');
    subplot(4,3,12),plot(tt,jzz);title('y jerk');
end


function polys = minimum_snap_single_axis_simple(waypts,ts,n_order,v0,a0,ve,ae)
p0 = waypts(1);
pe = waypts(end);

n_poly = length(waypts)-1;
n_coef = n_order+1;

% compute Q
Q_all = [];
for i=1:n_poly
    Q_all = blkdiag(Q_all,computeQ(n_order,3,ts(i),ts(i+1)));
end
b_all = zeros(size(Q_all,1),1);

Aeq = zeros(4*n_poly+2,n_coef*n_poly);
beq = zeros(4*n_poly+2,1);

% start/terminal pva constraints  (6 equations)
Aeq(1:3,1:n_coef) = [calc_tvec(ts(1),n_order,0);
                     calc_tvec(ts(1),n_order,1);
                     calc_tvec(ts(1),n_order,2)];
Aeq(4:6,n_coef*(n_poly-1)+1:n_coef*n_poly) = ...
                    [calc_tvec(ts(end),n_order,0);
                     calc_tvec(ts(end),n_order,1);
                     calc_tvec(ts(end),n_order,2)];
beq(1:6,1) = [p0,v0,a0,pe,ve,ae]';

% mid p constraints    (n_ploy-1 equations)
neq = 6;
for i=1:n_poly-1
    neq=neq+1;
    Aeq(neq,n_coef*i+1:n_coef*(i+1)) = calc_tvec(ts(i+1),n_order,0);
    beq(neq) = waypts(i+1);
end

% continuous constraints  ((n_poly-1)*3 equations)
for i=1:n_poly-1
    tvec_p = calc_tvec(ts(i+1),n_order,0);
    tvec_v = calc_tvec(ts(i+1),n_order,1);
    tvec_a = calc_tvec(ts(i+1),n_order,2);
    neq=neq+1;
    Aeq(neq,n_coef*(i-1)+1:n_coef*(i+1))=[tvec_p,-tvec_p];
    neq=neq+1;
    Aeq(neq,n_coef*(i-1)+1:n_coef*(i+1))=[tvec_v,-tvec_v];
    neq=neq+1;
    Aeq(neq,n_coef*(i-1)+1:n_coef*(i+1))=[tvec_a,-tvec_a];
end

Aieq = [];
bieq = [];

p = quadprog(Q_all,b_all,Aieq,bieq,Aeq,beq);

polys = reshape(p,n_coef,n_poly);

end

function ts = arrangeT(waypts,T)
    x = waypts(:,2:end) - waypts(:,1:end-1);
    dist = sum(x.^2,1).^0.5;
    k = T/sum(dist);
    ts = [0 cumsum(dist*k)];
end

function vals = polys_vals(polys,ts,tt,r)
    idx = 1;
    N = length(tt);
    vals = zeros(1,N);
    for i = 1:N
        t = tt(i);
        if t<ts(idx)
            vals(i) = 0;
        else
            while idx<length(ts) && t>ts(idx+1)+0.0001
                idx = idx+1;
            end
            vals(i) = poly_val(polys(:,idx),t,r);
        end
    end
end

function val = poly_val(poly,t,r)
    val = 0;
    n = length(poly)-1;
    if r<=0
        for i=0:n
            val = val+poly(i+1)*t^i;
        end
    else
        for i=r:n
            a = poly(i+1)*prod(i-r+1:i)*t^(i-r);
            val = val + a;
        end
    end
end

% n:polynormial order
% r:derivertive order, 1:minimum vel 2:minimum acc 3:minimum jerk 4:minimum snap
% t1:start timestamp for polynormial
% t2:end timestap for polynormial
function Q = computeQ(n,r,t1,t2)
    T = zeros((n-r)*2+1,1);
    for i = 1:(n-r)*2+1
        T(i) = t2^i-t1^i;
    end
    Q = zeros(n);
    for i = r+1:n+1
        for j = i:n+1
            k1 = i-r-1;
            k2 = j-r-1;
            k = k1+k2+1;
            Q(i,j) = prod(k1+1:k1+r)*prod(k2+1:k2+r)/k*T(k);
            Q(j,i) = Q(i,j);
        end
    end
end

    % r=0:pos  1:vel  2:acc 3:jerk
function tvec = calc_tvec(t,n_order,r)
    tvec = zeros(1,n_order+1);
    for i=r+1:n_order+1
        tvec(i) = prod(i-r:i-1)*t^(i-r-1);
    end
end