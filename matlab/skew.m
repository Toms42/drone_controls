function S = skew(x)
    % Chin 2013, p 133.
    % Defined as S(x)*y = cross(x,y)
    S = [0 -x(3) x(2) ; x(3) 0 -x(1) ; -x(2) x(1) 0 ];
end