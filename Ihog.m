%%Read images and create HOG feature matrix
function [ features ] = Ihog(imageList)
%fid = fopen('English\Hnd\all.txt~');
%line = fgetl(fid);
%features = [];
%count = 1;
%while (line ~= -1)
for i = 1:length(imageList)
    I = imresize(   rgb2gray(   imread( strcat(strcat('English\Hnd\',strrep(imageList(i,:),'/','\')),'.png' ))   ),  1/6);
    features(i,:) = extractHOGFeatures(I);    
%    line = fgetl(fid);
%    count = count + 1;
end

end