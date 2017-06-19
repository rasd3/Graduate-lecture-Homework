N = 5000;

A = [0 3/4 1/4 0;
     2/3 0 0 1/3;
     1/4 1/4 1/2 0;
     0 0 3/5 2/5];
Zt = [-2;-1;1;2];
sigma = 0.5;
Z = zeros(1, N);
Z(1, 1) = randi(4);
for i = 2:N
    Z(1, i) = find(rand<cumsum(A(Z(1,i-1),:)),1);
end
for i = 2:N
    Z(1, i)= Zt(Z(1, i));
end
Y = Z + normrnd(0,sigma,[1 N]);
O = cell(1);
O{1} = Y;
[model, likelihood] = HMM_EM(O,4);
