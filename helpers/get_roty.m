function[roty] = get_roty(pitch)  % theta
    roty = [cos(pitch), 0, sin(pitch);
             0, 1, 0;
             -sin(pitch), 0, cos(pitch)];
end