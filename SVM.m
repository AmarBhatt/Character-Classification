%% SVM
function [ classifier ] = SVM(features,labels)

%train svm

classifier = fitcecoc(features, labels);



