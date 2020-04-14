function[rotz] = get_rotz(yaw)  % psi
    rotz = [cos(yaw), -sin(yaw), 0;
             sin(yaw), cos(yaw), 0;
             0, 0, 1];
end