% Take only the first 100 examples to perform the binary classifcation "setosa vs versicolor" as
% Find the LDA projection vector w and calculate the discriminant for all training examples in
% MATLAB. Plot the histogram of the discriminant using the following code

%% data load

load fisheriris

x = meas(1:100, 1:2);
y = categorical(species(1:100));
labels = categories(y);
X1 = x(y==labels{1}, :);
X2 = x(y==labels{2}, :);

%% LDA proejction

% number of observations of each class
N1 = size(X1, 1);
N2 = size(X2, 1);
N = N1 + N2;

% mean of each class
mu1 = mean(X1);
mu2 = mean(X2);

% average of the mean of all classes
mu = (mu1*(N1/N)+mu2*(N2/N))/2;

% center the data
d1 = X1 - repmat(mu1, N1, 1);
d2 = X2 - repmat(mu2, N2, 1);

% Sw
s1 = d1'*d1;
s2 = d2'*d2;
sw = s1+s2;

% Sb
sb1 = N1*(mu1-mu)'*(mu1-mu);
sb2 = N2*(mu2-mu)'*(mu2-mu);
sb = sb1+sb2;

% same as inv(sw)*(mu1-muw2)
v = sw\sb;
[evec, eval] = eig(v);

% project the data of the first and second class respectively
y1 = X1*evec(:,2);
y2 = X2*evec(:,2);

%% figure
figure;
range = min(vertcat(y1,y2)):0.1:max(vertcat(y1,y2));
[cnt1, xbin1] = hist(y1, range);
[cnt2, xbin2] = hist(y2, range);
bar(xbin1, cnt1, 'r');
hold on;
bar(xbin2, cnt2, 'b')
