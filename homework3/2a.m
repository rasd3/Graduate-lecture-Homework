clear;

A = [0 3/4 1/4 0;
     2/3 0 0 1/3;
     1/4 1/4 1/2 0;
     0 0 3/5 2/5];
Zt = [-2;-1;1;2];
err = zeros(1,31);
cnt = 1;


for SIG = -40:2:20
    disp(SIG)
    sigma = sqrt(10^(SIG/-10));
    for step = 1:1000
        % initialize
        alpha = zeros(4,100);
        beta = zeros(4,100);
        B = zeros(4,100);
        Z = zeros(1,100);
        Z(1, 1) = randi(4);
        for i = 2:100
            Z(1, i) = find(rand<cumsum(A(Z(1,i-1),:)),1);
        end

        % calculate B matrix
        Y = Z + normrnd(0,sigma,[1 100]);
        for i = 1:100
            B(:,i) = normpdf(Y(i),Zt',sigma);
        end
        
        % evaluate alpha & beta & gamma by forward, backward recursion
        alpha(:,1) = 0.25.*B(:,1);
        beta(:,100) = 1;
       
        for t = 1:99
            alpha(:,t+1) = alpha(:,t)'*A.*B(:,t+1)';
        end
        for t = 100:-1:2
            beta(:,t-1) = sum(A.*repmat(B(:,t)',4,1).*repmat(beta(:,t)',4,1),2);
        end
        gamma = (alpha.*beta)./repmat(sum(alpha.*beta),4,1);
        
        % MAP estimate
        [~, Zh] = max(gamma);
        tf = sum(eq(Zh,Z));
        tf = 100 - tf;
        
        err(cnt) = err(cnt) + tf/1000;
    end
    disp(err(cnt));
    cnt = cnt+1;
end


figure;
plot(-40:2:20,err);
