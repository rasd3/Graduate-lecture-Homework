
% (c)
% Run SVM model using fitcsvm() in MATLAB. (By default, SVM model uses linear kernel).
% Calculate the discriminant for all training examples. Plot the histogram of the discriminant.

% (d)
% Construct the target vector whose element has 1 for setosa class and 2 for versicolor. Evaluate
% the relevance of the discriminant with the target vector using corrcoef function. (We can use
% correlation coeffcient between the discriminant and the target to indicate the quality of the
% discriminant)

% ref: https://kr.mathworks.com/help/stats/fitcsvm.html

%% data load
load fisheriris
inds = ~strcmp(species,'virginica');
X = meas(inds,3:4);
y = species(inds);

%% SVM 
% initialize & training
SVMModel = fitcsvm(X,y)
classOrder = SVMModel.ClassNames

%% figure

sv = SVMModel.SupportVectors;
figure
gscatter(X(:,1),X(:,2),y)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('versicolor','virginica','Support Vector')
hold off

%% class loss

CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)

%% check correlation coefficient between the discriminant and the target vector

pred = categorical(kfoldPredict(CVSVMModel));
y = categorical(y);

pred_setosa = pred=='setosa';
pred_versicolor = pred=='versicolor';
pred_ = pred_setosa + pred_versicolor*2;
gt_setosa = y=='setosa';
gt_versicolor = y=='versicolor';
gt = gt_setosa + gt_versicolor*2;

corrcoef(pred_, gt)

