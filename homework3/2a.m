A = [0 3/4 1/4 1;
     2/3 1 1 1/3;
     1/4 1/4 1/2 1;
     0 0 3/5 2/5];
A = cumsum(A);
Zt = [-2;1;1;2];


for SIG = -10:0.5:40
    sigma = 10^(SIG/-20);
    for step = 1:1000
        % initialize
        alpha = zeros(1,100);
        beta = zeros(1,100);
        gamma = zeros(1,100);
        B = zeros(4,100);
        Z = zeros(1,100);
        Z(1) = randi(4);
        for i = 2:100
            Z(i) = 1 + sum(rand() > A(Z(i-1),:));
        end

        % calculate B matrix
        Y = Z + normrnd(0,sigma^2,[1 100]);
        for i = 1:4
            B(i,:) = normpdf(Y,Zt(i),sigma^2);
        end
        
        % evaluate alpha & betta & gamma
        alpha(1) = 0.25*B(Z(1),1);
        for t = 1:99
            alpha(t+1) = 0;
            for i = 1:4
                alpha(t+1) = alpha(t+1) + alpha(t)*A(Z(t),Z(t+1))*B(Z(t+1),t+1);
            end 
        end
        betta(100) = 1;
        for t = 100:-1:2
            beta(t-1) = 0;
            for i = 1:4
                beta(t-1) = beta(t-1) + A(Z(t-1),Z(t))*B(Z(t),t)*beta(t);
            end
        end
        sum = 0;
        for t = 1:100
            sum = sum + alpha(t)*beta(t);
            gamma(t) = alpha(t)*beta(t);
        end
        gamma = gamma./sum;
        
        % MAP estimate
        [sdf, Zh] = max(gamma);
        tf = eq(Zh,Z);
    end
end
