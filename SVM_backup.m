%% Main

%load lists
display('1a. Load Lists');
list_English_Hnd;
names = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
display('1b. Finish Load Lists');

%get hog for training Images
display('2a. Get HOG For Training');
trainImages = list.ALLnames(list.TRNind(:,end),:);
trainFeatures = Ihog(trainImages);
display('2b. Finish Get HOG For Training');

%train svm
display('3a. Train Classifier');
classifier = fitcecoc(trainFeatures, list.ALLlabels(list.TRNind(:,end),:));
display('3b. Finish Train Classifier');

%get hog for test images
display('4a. Get HOG for testing');
testImages = list.ALLnames(list.TSTind(:,end),:);
testFeatures = Ihog(testImages);
display('4b. Finish Get HOG for testing');

%Predict
display('5a. Predicting');
prediction = predict(classifier, testFeatures);
display('5b. Finish Predicting');

%Confusion Matrix
display('6a. Confusion Matrix');
%confMat = confusionmat(list.ALLlabels(list.TSTind(:,end),:), prediction);
figure;
C = plotConfusionMatrix(list.ALLlabels(list.TSTind(:,end),:),prediction,names);
figure;
surf(C);
display('6b. Finish Confusion Matrix');
%helperDisplayConfusionMatrix(confMat);
