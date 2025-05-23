% DEI Practical assignment 2014/15
% Objective: 3D HCI Gesture learning

%===========================================================================

%=====================
%| - ROI DETECTION - |
%=====================

% LOAD DATA

% scan3d.img = [width_pixel, height_pixel, RGB_color, frame]

load("./Secuencias/scan3d-bg-27Feb2014-094402.mat");

scanWithNans = single(scan3d.depth);

scanWithNans(scanWithNans==0) = NaN;  %0 to Nans

RColor(:,:,:) = scan3d.img(:,:,1,:);
GColor(:,:,:) = scan3d.img(:,:,2,:);
BColor(:,:,:) = scan3d.img(:,:,3,:);

meanScene = median(scanWithNans,3);
meanSceneColor(:,:,1) = median(RColor,3);
meanSceneColor(:,:,2) = median(GColor,3);
meanSceneColor(:,:,3) = median(BColor,3);
meanSceneColor = single(meanSceneColor);

desvScene = std(scanWithNans,0,3);
desvSceneColor(:,:,1) = std(single(RColor),0,3);
desvSceneColor(:,:,2) = std(single(GColor),0,3);
desvSceneColor(:,:,3) = std(single(BColor),0,3);
desvSceneColor = single(desvSceneColor);

meanScene(isnan(meanScene)) = 0;  %Nans to 0
%______________________________

load("./Secuencias/scan3d-o-27Feb2014-093907.mat");
%load("./Secuencias/scan3d-o-27Feb2014-093946.mat");
%load("./Secuencias/scan3d-o-27Feb2014-094033.mat");
%load("./Secuencias/scan3d-ri-27Feb2014-094457.mat");
%load("./Secuencias/scan3d-ri-27Feb2014-094528.mat");
%load("./Secuencias/scan3d-ri-27Feb2014-094558.mat");
%load("./Secuencias/scan3d-fw-27Feb2014-094714.mat");
%load("./Secuencias/scan3d-fw-27Feb2014-094834.mat");
%load("./Secuencias/scan3d-fw-27Feb2014-094752.mat");
%load("./Secuencias/scan3d-up-27Feb2014-094145.mat");
%load("./Secuencias/scan3d-up-27Feb2014-094221.mat");
%load("./Secuencias/scan3d-up-27Feb2014-094258.mat");

numFrames = size(scan3d.img,4);

depthWithNans = single(scan3d.depth);
depthWithNans(depthWithNans==0) = NaN;  %0 to Nans

RColor(:,:,:) = scan3d.img(:,:,1,:);
GColor(:,:,:) = scan3d.img(:,:,2,:);
BColor(:,:,:) = scan3d.img(:,:,3,:);

%===========================================================================

% SEGMENTATION

% Background segmentation and RGBD mask

maskAuxD = createMaskWithBS(depthWithNans, meanScene, desvScene, 8);
maskAuxR = createMaskWithBS(RColor, meanSceneColor(:,:,1), desvSceneColor(:,:,1), 2);
maskAuxG = createMaskWithBS(GColor, meanSceneColor(:,:,2), desvSceneColor(:,:,2), 1);
maskAuxB = createMaskWithBS(BColor, meanSceneColor(:,:,3), desvSceneColor(:,:,3), 1);

for i=1 : numFrames
    maskColor(:,:,i) = maskAuxR(:,:,i) & maskAuxG(:,:,i) & maskAuxB(:,:,i);
    maskColorDepth(:,:,i) = maskColor(:,:,i) & depthWithNans(:,:,i)<1600;
end

RSegmented = RColor*NaN;
GSegmented = GColor*NaN;
BSegmented = BColor*NaN;
DSegmented = scan3d.depth*NaN;
RGBSegmented = scan3d.img*0;

for i=1 : numFrames
    RSegmented(:,:,i) = segmentImageByColorMask(RColor, maskColorDepth, i);
    GSegmented(:,:,i) = segmentImageByColorMask(GColor, maskColorDepth, i);
    BSegmented(:,:,i) = segmentImageByColorMask(BColor, maskColorDepth, i);
    DSegmented(:,:,i) = segmentImageByColorMask(depthWithNans, maskColorDepth, i);

    RGBSegmented(:,:,1,i) = uint8(RSegmented(:,:,i));
    RGBSegmented(:,:,2,i) = uint8(GSegmented(:,:,i));
    RGBSegmented(:,:,3,i) = uint8(BSegmented(:,:,i));
end
%______________________________

% Centroid

%bbs = [];

for i=1 : numFrames
    % RGB to HSV
    HSVimage = rgb2hsv(RGBSegmented(:,:,:,i));
    maskSkin(:,:,i) = HSVimage(:,:,3)>0.6;

    % Find unique objects in binary segmented image
    regions = regionprops(maskSkin(:,:,i));

    % We want the biggets objects
    reg = [];
    for j=1 : length(regions)
        reg(j) = regions(j).Area;
    end
    [b,idx] = sort(reg, "descend");

    % Blobs
    bb1 = round(regions(idx(1)).BoundingBox); %should be the hand
    bb2 = round(regions(idx(2)).BoundingBox); %should be the face

    % Crop regions within image boundaries
    bbc1 = imcrop(DSegmented(:,:,i), bb1);
    bbc2 = imcrop(DSegmented(:,:,i), bb2);

    % Mean depth of blobs
    valuesReg1 = mean(bbc1(:));
    valuesReg2 = mean(bbc2(:));

    % Avoid noise
    if reg(idx(2))<500
        valuesReg2 = inf;
    end

    % We want the closest Centroid
    if valuesReg1<valuesReg2
        Centroid(:,i) = regions(idx(1)).Centroid;
        %bbs = [bbs; bb1];
    else
        Centroid(:,i) = regions(idx(2)).Centroid;
        %bbs = [bbs; bb2];
    end
end

%fid = fopen("./blobs/scan3d-o-27Feb2014-093907.mat", "w");
%fprintf(fid, "[%d,%d,%d,%d], ", bbs');
%fclose(fid);

%===========================================================================

%================
%| - Tracking - |
%================

% Kalman filter
kalmanFilter = trackingKF( ...
    "StateTransitionModel", eye(2), ...
    "MeasurementModel", eye(2), ...
    "StateCovariance", eye(2) * 1000 ...
);

predicted = zeros(2, numFrames);
location = zeros(2, numFrames);

for i=1 : numFrames
    predicted(:,i) = predict(kalmanFilter);

    % Correct centroid
    measured = Centroid(:,i);
    location(:,i) = correct(kalmanFilter, measured);
end

%===========================================================================

%=====================
%| - Clasification - |
%=====================

data = readmatrix("data.csv", "OutputType", "string");

% Support Vector Machines Model
c = cvpartition(data(:,3), "KFold", 3);
cvSVMModel = fitcecoc( ...
    double(data(:,1:2)), data(:,3), ...
    "CVPartition", c, "Learners", "svm"...
);

% Find the best model with Cross Validation
L = kfoldLoss(cvSVMModel, "Mode", "individual");
[minError, minIndex] = min(L);
SVMModel = cvSVMModel.Trained{minIndex};
save("model.mat", "SVMModel");

% See model performance
labels = predict(SVMModel,cvSVMModel.X);
cm = confusionchart(data(:,3),string(labels), ...
    "Title","Matriz de confusión","ColumnSummary","column-normalized", ...
    "RowSummary","row-normalized","Normalization","absolute");
cvSVMMdlError = kfoldLoss(cvSVMModel); % error of crossvalidated model


% Extract characteristics
xmov = 0;
ymov = 0;
for i=2 : numFrames
    xmov = xmov + abs(location(1,i) - location(1,i-1));
    ymov = ymov + abs(location(2,i) - location(2,i-1));
end
mov = [xmov, ymov];

%data = [data; [mov, "o"]];
%writematrix(data, "data.csv");

% Predict
label = predict(SVMModel, mov);

%===========================================================================

% Display

fig2 = figure;
for i=1 : numFrames
    imagesc(scan3d.img(:,:,:,i)); %Original
    imagesc(DSegmented(:,:,i)); %Segmented

    hold on
    plot(location(1,i), location(2,i), "r.", "markersize", 50); %Centroid
    hold off
    pause(0.1);
end
hold on
plot(location(1,:), location(2,:), "r-"); %Centroid history
text(50,50, label, "Color", "red", "FontSize", 20, "FontWeight", "bold");
hold off