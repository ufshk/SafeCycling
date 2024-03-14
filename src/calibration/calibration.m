images = imageDatastore("src/calibration/chessboard/bitmaps/");
imageFileNames = images.Files;

[imagePoints,boardSize,imagesUsed] = detectCheckerboardPoints(imageFileNames,HighDistortion=true);
usedImageFileNames = imageFileNames(imagesUsed);
images = imageDatastore(usedImageFileNames);

squareSize = 28; % millimeters
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

I = readimage(images,9); 
imageSize = [size(I,1) size(I,2)];
params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize);

figure
showReprojectionErrors(params);

figure
showExtrinsics(params);

drawnow

figure 
imshow(I); 
hold on
plot(imagePoints(:,1,9),imagePoints(:,2,9),"go");
plot(params.ReprojectedPoints(:,1,9),params.ReprojectedPoints(:,2,9),"r+");
legend("Detected Points","Reprojected Points");
hold off