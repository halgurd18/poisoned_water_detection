% clear all
% clc
% close all
%%%%%%%%%%
tic
load datam
load labels.mat
% ToxicWater=datam(1:1075,:);
% CleanWater=datam(2823:end,:);
CleanWater=datam(1:2822,:);
ToxicWater=datam(2823:end,:);
[Avg_std_ACC ,Avg_std_AUC, Avg_std_TPR,Avg_Std_TNR,Avg_std_F1Score, confuMatrices,models]=Binary_SVM_optimised(CleanWater,ToxicWater);

toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Avg_std_ACC ,Avg_std_AUC, Avg_std_TPR,Avg_Std_TNR,Avg_std_F1Score, confuMatrices,models]=Binary_SVM_optimised(CleanWater,ToxicWater)

%%%%%% Classification Stage %%%%%%%%%
%%% CW=Clean Water, TW=Toxic Water

%%% 5 Fold cross Validation
Fold_Num=5;
CW_size=size(ToxicWater,1);
TW_size=size(CleanWater,1);
for i=1:1 % this for loop if we want to repeat the 5FCV more than once
    rng(i)% For reproducibility 
r1 = randperm(CW_size);% Randomly generate indecies
r2 = randperm(TW_size);% radnomly generate indices
ToxicWater=ToxicWater(r1,:);%re-arranges the indices of TW accordingly
CleanWater=CleanWater(r2,:);%re-arrange CW indices accodingly
%partition/folds exact number of samples according to each class
CW_par=floor(CW_size/Fold_Num); 
TW_par=floor(TW_size/Fold_Num);
% Select the 5 folds(i.e paritions) from each class
p1_lig=ToxicWater(1:CW_par,:);
p2_lig=ToxicWater(CW_par+1:2*CW_par,:);
p3_lig=ToxicWater((2*CW_par)+1:(3*CW_par),:);
p4_lig=ToxicWater((3*CW_par)+1:(4*CW_par),:);
p5_lig=ToxicWater((4*CW_par)+1:CW_size,:);
p1_dec=CleanWater(1:TW_par,:);
p2_dec=CleanWater(TW_par+1:2*TW_par,:);
p3_dec=CleanWater((2*TW_par)+1:(3*TW_par),:);
p4_dec=CleanWater((3*TW_par)+1:(4*TW_par),:);
p5_dec=CleanWater((4*TW_par)+1:TW_size,:);
% Training data preparation and label generation for each fold
training_data1=[[p2_lig;p3_lig;p4_lig;p5_lig],zeros(size([p2_lig;p3_lig;p4_lig;p5_lig],1),1);[p2_dec;p3_dec;p4_dec;p5_dec],ones(size([p2_dec;p3_dec;p4_dec;p5_dec],1),1)];
training_data2=[[p1_lig;p3_lig;p4_lig;p5_lig],zeros(size([p1_lig;p3_lig;p4_lig;p5_lig],1),1);[p1_dec;p3_dec;p4_dec;p5_dec],ones(size([p1_dec;p3_dec;p4_dec;p5_dec],1),1)];
training_data3=[[p1_lig;p2_lig;p4_lig;p5_lig],zeros(size([p1_lig;p2_lig;p4_lig;p5_lig],1),1);[p1_dec;p2_dec;p4_dec;p5_dec],ones(size([p1_dec;p2_dec;p4_dec;p5_dec],1),1)];
training_data4=[[p1_lig;p2_lig;p3_lig;p5_lig],zeros(size([p1_lig;p2_lig;p3_lig;p5_lig],1),1);[p1_dec;p2_dec;p3_dec;p5_dec],ones(size([p1_dec;p2_dec;p3_dec;p5_dec],1),1)];
training_data5=[[p1_lig;p2_lig;p3_lig;p4_lig],zeros(size([p1_lig;p2_lig;p3_lig;p4_lig],1),1);[p1_dec;p2_dec;p3_dec;p4_dec],ones(size([p1_dec;p2_dec;p3_dec;p4_dec],1),1)];
% Testing data preparation and concatenation for each of the folds
testing_data1=[p1_lig;p1_dec];
testing_data2=[p2_lig;p2_dec];
testing_data3=[p3_lig;p3_dec];
testing_data4=[p4_lig;p4_dec];
testing_data5=[p5_lig;p5_dec];
% Training SVM model and optimizing all of its hyperparameters.
SVMModel1 = fitcsvm(training_data1(:,1:end-1),training_data1(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel2 = fitcsvm(training_data2(:,1:end-1),training_data2(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel3 = fitcsvm(training_data3(:,1:end-1),training_data3(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel4 = fitcsvm(training_data4(:,1:end-1),training_data4(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
SVMModel5 = fitcsvm(training_data5(:,1:end-1),training_data5(:,end),'Standardize',true,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));


%%%% Ensemble Classifier If we want to use it later
% t = templateTree('MaxNumSplits',5);
% Mdl1 = fitcensemble(training_data1(:,1:end-1),training_data1(:,end),'OptimizeHyperparameters','all','Learners',t, ...
%     'HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
% Mdl2 = fitcensemble(training_data2(:,1:end-1),training_data2(:,end),'OptimizeHyperparameters','all','Learners',t, ...
%     'HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
% Mdl3 = fitcensemble(training_data3(:,1:end-1),training_data3(:,end),'OptimizeHyperparameters','all','Learners',t, ...
%     'HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
% 
% Mdl4 = fitcensemble(training_data4(:,1:end-1),training_data4(:,end),'OptimizeHyperparameters','all','Learners',t, ...
%     'HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));
% 
% Mdl5 = fitcensemble(training_data5(:,1:end-1),training_data5(:,end),'OptimizeHyperparameters','all','Learners',t, ...
%     'HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',0));

%%% Prediction stage for each of the models
[fold1_label,fold1_score] = predict(SVMModel1,testing_data1);
[fold2_label,fold2_score] = predict(SVMModel2,testing_data2);
[fold3_label,fold3_score] = predict(SVMModel3,testing_data3);
[fold4_label,fold4_score] = predict(SVMModel4,testing_data4);
[fold5_label,fold5_score] = predict(SVMModel5,testing_data5);
%%% Confusion matrix calculation as well as testing label generation
C1 = confusionmat(fold1_label,[zeros(size(p1_lig,1),1);ones(size(p1_dec,1),1)]);
C2 = confusionmat(fold2_label,[zeros(size(p2_lig,1),1);ones(size(p2_dec,1),1)]);
C3 = confusionmat(fold3_label,[zeros(size(p3_lig,1),1);ones(size(p3_dec,1),1)]);
C4 = confusionmat(fold4_label,[zeros(size(p4_lig,1),1);ones(size(p4_dec,1),1)]);
C5 = confusionmat(fold5_label,[zeros(size(p5_lig,1),1);ones(size(p5_dec,1),1)]);
% computing AUC number
[~,~,~,AUC1,~] = perfcurve([zeros(size(p1_lig,1),1);ones(size(p1_dec,1),1)],fold1_score(:,1),0);
[~,~,~,AUC2,~] = perfcurve([zeros(size(p2_lig,1),1);ones(size(p2_dec,1),1)],fold2_score(:,1),0);
[~,~,~,AUC3,~] = perfcurve([zeros(size(p3_lig,1),1);ones(size(p3_dec,1),1)],fold3_score(:,1),0);
[~,~,~,AUC4,~] = perfcurve([zeros(size(p4_lig,1),1);ones(size(p4_dec,1),1)],fold4_score(:,1),0);
[~,~,~,AUC5,~] = perfcurve([zeros(size(p5_lig,1),1);ones(size(p5_dec,1),1)],fold5_score(:,1),0);
% If we want to save some of the results
AUC_sets(i,:)=[AUC1,AUC2,AUC3,AUC4,AUC5];
confuMatrices=[C1,C2,C3,C4,C5];
ConfMats_sets{i,1}=confuMatrices;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % the positive class is TP therefore the confusion matrix terminology is
% % as follows: Positive class means Poison, negative class means
% % clean . In other words, TN means the water is negative of poison (=CW)
% %    TN  FP
% %    FN  TP
%%%sensitivity, recall, hit rate, or true positive rate (TPR)
%%% TPR=TP/(TP+FN)
%%%% specificity, selectivity or true negative rate (TNR)
%%% TNR=TN/(TN+FP)
%%%% Accuracy
%%% Acc=(TN+TP)/(TN+TP+FN+FP)
%%%%% F1-Score
%%% F1-score= (2*TP)/(2*TP+FP+FN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fold1_acc=(C1(1,1)+C1(2,2))/(C1(1,1)+C1(2,2)+C1(1,2)+C1(2,1));
fold2_acc=(C2(1,1)+C2(2,2))/(C2(1,1)+C2(2,2)+C1(1,2)+C2(2,1));
fold3_acc=(C3(1,1)+C3(2,2))/(C3(1,1)+C3(2,2)+C1(1,2)+C3(2,1));
fold4_acc=(C4(1,1)+C4(2,2))/(C4(1,1)+C4(2,2)+C1(1,2)+C4(2,1));
fold5_acc=(C5(1,1)+C5(2,2))/(C5(1,1)+C5(2,2)+C1(1,2)+C5(2,1));

fold1_tpr=C1(2,2)/(C1(2,2)+C1(2,1));
fold2_tpr=C2(2,2)/(C2(2,2)+C2(2,1));
fold3_tpr=C3(2,2)/(C3(2,2)+C3(2,1));
fold4_tpr=C4(2,2)/(C4(2,2)+C4(2,1));
fold5_tpr=C5(2,2)/(C5(2,2)+C5(2,1));

fold1_tnr=C1(1,1)/(C1(1,1)+C1(1,2));
fold2_tnr=C2(1,1)/(C2(1,1)+C2(1,2));
fold3_tnr=C3(1,1)/(C3(1,1)+C3(1,2));
fold4_tnr=C4(1,1)/(C4(1,1)+C4(1,2));
fold5_tnr=C5(1,1)/(C5(1,1)+C5(1,2));

fold1_f1Score=(2*C1(2,2))/((2*C1(2,2))+C1(1,2)+C1(2,1));
fold2_f1Score=(2*C2(2,2))/((2*C2(2,2))+C2(1,2)+C2(2,1));
fold3_f1Score=(2*C3(2,2))/((2*C3(2,2))+C3(1,2)+C3(2,1));
fold4_f1Score=(2*C4(2,2))/((2*C4(2,2))+C4(1,2)+C4(2,1));
fold5_f1Score=(2*C5(2,2))/((2*C5(2,2))+C5(1,2)+C5(2,1));

Fold_acc=[fold1_acc,fold2_acc,fold3_acc,fold4_acc,fold5_acc];
Fold_TPR=[fold1_tpr,fold2_tpr,fold3_tpr,fold4_tpr,fold5_tpr];
Fold_FPR=[fold1_tnr,fold2_tnr,fold3_tnr,fold4_tnr,fold5_tnr];
Fold_F1_score=[fold1_f1Score,fold2_f1Score,fold3_f1Score,fold4_f1Score,fold5_f1Score];
%%%% To keep track of the parameters for each of the folds, we can/will
%%%% save the trained models
models={SVMModel1,SVMModel2,SVMModel3,SVMModel4,SVMModel5};
i
end
Avg_std_ACC=[mean(Fold_acc), std(Fold_acc)];
Avg_std_AUC=[mean(AUC_sets),std(AUC_sets)];
Avg_std_TPR=[mean(Fold_TPR), std(Fold_TPR)];
Avg_Std_TNR=[mean(Fold_FPR), std(Fold_FPR)];
Avg_std_F1Score=[mean(Fold_F1_score), std(Fold_F1_score)];
end