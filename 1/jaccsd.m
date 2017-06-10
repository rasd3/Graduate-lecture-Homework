function [z,A]=jaccsd(fun,x,K)
% JACCSD Jacobian through complex step differentiation
% [z J] = jaccsd(f,x)
% z = f(x)
% J = f'(x)
%
if nargin == 3, z = fun(x,K); end
if nargin == 2, z = fun(x); end
n=numel(x);
m=numel(z);
A=zeros(m,n);
h=n*eps;
for k=1:n
    x1=x;
    x1(k)=x1(k)+h*k;
    if nargin == 3, A(:,k) = imag(fun(x1,K))/h; end;
    if nargin == 2, A(:,k) = imag(fun(x1))/h; end;
end
end
