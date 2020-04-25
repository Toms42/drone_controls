function dp = ppder(p)
%PPDER Piecewise Differentiate polynomial.
%   PPDER(P) returns the derivative of the polynomial whose
%   coefficients are the elements of vector P like polyder.
%   if P is a struct determined for example by mkpp,
%   P is the derivative of the piecewise  polynom.
%   AAA A AMELIORER COMME POLYDER ???
%
%
%   Class support for inputs u, v:
%      float: double, single
%
%   See also POLYINT
%
%   Jérôme Bastien 2014/04/08
%   jerome.bastien@univ-lyon1.fr
%   http://utbmjb.chez-alice.fr/

if ~isstruct(p)
    dp=polyder(p);
else
    P=p.coefs;
    T=size(P);
    Q=T(1);
    n=T(2);
    if n==1
        dpp=zeros(Q,1);
    else
        R=n-1:-1:1;
        R=R(ones(1,Q),:);
        dpp=P(:,1:end-1).*R;
    end
    dp=mkpp(p.breaks,dpp);
end