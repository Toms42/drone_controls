function [] = plotpp(pp, lim)
%PLOTPP plots a pp
x = lim(1):0.01:lim(2);
y = ppval(pp, x);
plot(x,y);
end

