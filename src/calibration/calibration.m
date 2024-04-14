images = imageDatastore("chessboard/bitmaps/");
imageFileNames = images.Files;

[imagePoints,boardSize,imagesUsed] = detectCheckerboardPoints(imageFileNames,HighDistortion=true);
usedImageFileNames = imageFileNames(imagesUsed);
images = imageDatastore(usedImageFileNames);

squareSize = 28; % millimeters
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

I = readimage(images,9); 
imageSize = [size(I,1) size(I,2)];
params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize);

% msg = rosmessage("sensor_msgs/CameraInfo","DataFormat","struct");
% msg = rosWriteCameraInfo(msg,toStruct(params));