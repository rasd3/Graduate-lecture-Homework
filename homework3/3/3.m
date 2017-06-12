%% (a) generate 2000 GMM data

N = 4000;
Mu = [2 2; -2 -1; 1 -2];
Sigma = cat(3, [1 0;0 2], [1 0.5;0.5 2], [0.5 1/3;1/3 1]);
P = [2/9 1/3 4/9];
gm = gmdistribution(Mu,Sigma,P);

data = random(gm, N);
scatter(data(:,1),data(:,2));

%% (b) run bayesian inference algorithm

figure;
gamma = zeros(N, 3);
rgb = [[1 0 0];[0 0 1];[0 1 0]];
for j = 1:3
    gamma(:,j) = P(j)*mvnpdf(data,Mu(j,:),Sigma(:,:,j)); % multi-dimensional pdf
end
[a, color] = max(gamma,[],2);

scatter(data(:,1),data(:,2),25,rgb(color,:));

%% (c) run EM algorithm

figure;
Step = 200;
% expectations
eMu = [-2 4; 3 -2; 4 4];
eP = [1/3 1/3 1/3];
eSigma = cat(3, [1 0.5;0.5 1],[1 0.5;0.5 1],[1 0.5;0.5 1]);
% log-likelihood
logL = zeros(1,Step);

for s = 1:Step
    % E step
    for j = 1:3
        gamma(:,j) = eP(j)*mvnpdf(data,eMu(j,:),eSigma(:,:,j));
    end
    sumG = sum(gamma,2);
    gamma = gamma./sumG;
    logL(s) = log(sum(sumG,1));
    
    % M step
    eMu = zeros(3,2);
    eSigma = zeros(2,2,3);
    for j = 1:3
        Nk = sum(gamma(:,j),1);
        eMu(j,1) = sum(gamma(:,j).*data(:,1)/Nk);
        eMu(j,2) = sum(gamma(:,j).*data(:,2)/Nk);
        for i=1:N
            eSigma(:,:,j) = eSigma(:,:,j) + gamma(i,j).*(data(i,:)-eMu(j,:))'*(data(i,:)-eMu(j,:))/Nk;
        end
        eP(j) = Nk/N;
    end
end

plot(logL);
xlabel('Iteration');
ylabel('Observed data Log-likelihood');
