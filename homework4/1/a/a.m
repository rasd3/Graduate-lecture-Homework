Num = 1;
v = sqrt(10);
n = 1;
V = v^2*eye(Num);
N = n^2;            				% Cov of process & measurement
f = @(x,k)x/2+25*x/(1+x^2)+8*cos(1.2*k);	% Nonlinear state equation
h = @(x)x^2/20;					% Measurement Equation
s = 0;						% initial state
x = s+v*normrnd(0,10);				% initial state with noise
P = 1;						% initisl state covariance
N = 1000;					% total step
xV = zeros(Num, N);
rxV = zeros(Num, N);
rx = 0;
MSE = 0;
for k=1:N
    z = h(s) + n*randn;
    [x, P] = ekf(f,x,k,P,h,z,V,N);
    xV(k) = x;
    rxV(k) = f(rx,k);
    rx = f(rx,k);
    s = f(s,k) +n *randn;
    MSE = MSE +(rx-x)^2;
end
MSE = MSE/1000
plot(900:N,xV(900:N),'-')
hold on;
plot(900:N,rxV(900:N),'b--')
 
% Example:
%{
n=3;      %number of state
q=0.1;    %std of process 
r=0.1;    %std of measurement
Q=q^2*eye(n); % covariance of process
R=r^2;        % covariance of measurement  
f=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];  % nonlinear state equations
h=@(x)x(1);                               % measurement equation
s=[0;0;1];                                % initial state
x=s+q*randn(3,1); %initial state          % initial state with noise
P = eye(n);                               % initial state covraiance
N=20;                                     % total dynamic steps
xV = zeros(n,N);          %estmate        % allocate memory
sV = zeros(n,N);          %actual
zV = zeros(1,N);
for k=1:N
  z = h(s) + r*randn;                     % measurments
  sV(:,k)= s;                             % save actual state
  zV(k)  = z;                             % save measurment
  [x, P] = ekf(f,x,P,h,z,Q,R);            % ekf 
  xV(:,k) = x;                            % save estimate
  s = f(s) + q*randn(3,1);                % update process 
end
for k=1:3                                 % plot results
  subplot(3,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
end
%}
% By Yi Cao at Cranfield University, 02/01/2008
%
