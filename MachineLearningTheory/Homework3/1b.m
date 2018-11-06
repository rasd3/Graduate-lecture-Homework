% Find the PCA projection vector w and calculate the discriminant for all training examples in
% MATLAB (refer to Chapter 12.1 in PRML book). Plot the histogram of the discriminant.

%% data load

load fisheriris

x = meas(1:100, 1:2);
y = categorical(species(1:100));

labels = categories(y);
x1 = x(y==labels{1}, :);
x2 = x(y==labels{2}, :);

%% PCA

xmean = mean(meas(1:100, 1));
ymean = mean(meas(1:100, 2));
xnew = meas(1:100, 1)-xmean*ones(100 ,1);
ynew = meas(1:100, 2)-ymean*ones(100, 1);

% covariance matrix
covxy = cov(xnew, ynew);
[evec, eval] = eig(covxy);
eval = diag(eval);

% projection vector W
W = evec(:, find(eval==max(eval)))

% deriving the new data set
result = W'*[xnew, ynew]';

%% figure

result1 = result<0;
result2 = result>=0;
gscatter(x(:,1), x(:,2), result1'+result2'*2, 'rg', 'os');
