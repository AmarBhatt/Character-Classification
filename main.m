%%Load data

list_English_Hnd;
names = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
digit_names = {'0','1','2','3','4','5','6','7','8','9'};
%uppercase_names = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
%lowercase_names = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};

labels = list.ALLlabels(list.TRNind(:,end),:);
digit_labels = labels(1:150,:);
%uppercase_labels = labels(151:540,:);
%lowercase_labels = labels(541:930,:);

test_labels = list.ALLlabels(list.TSTind(:,end),:);
digit_test_labels = test_labels(1:270,:);
%uppercase_test_labels = test_labels(270:972,:);
%lowercase_test_labels = test_labels(973:1674,:);

trainImages = list.ALLnames(list.TRNind(:,end),:);
trainDigitImages = trainImages(1:150,:);
%trainUpperCaseImages = trainImages(151:540,:);
%trainLowerCaseImages = trainImages(541:930,:);

testImages = list.ALLnames(list.TSTind(:,end),:);
testDigitImages = testImages(1:270,:);
%testUpperCaseImages = testImages(270:972,:);
%testLowerCaseImages = testImages(973:1674,:);

trainDigitFeatures = Ihog(trainDigitImages);
%trainUpperCaseFeatures = Ihog(trainUpperCaseImages);
%trainLowerCaseFeatures = Ihog(trainLowerCaseImages);

testDigitFeatures = Ihog(testDigitImages);
%testUpperCaseFeatures = Ihog(testUpperCaseImages);
%testLowerCaseFeatures = Ihog(testLowerCaseImages);


%Digit
digit_classifier = SVM(testDigitFeatures,digit_test_labels);
digit_prediction = predict(digit_classifier, trainDigitFeatures);

figure;
C = plotConfusionMatrix(digit_labels,digit_prediction,digit_names,'hot',[0,1,2,3,4,5,6,7,8,9]);
figure;
surf(C);


