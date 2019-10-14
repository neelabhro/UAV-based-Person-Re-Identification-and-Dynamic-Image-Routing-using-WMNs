%Neelabhro Roy
%IIIT, Delhi

%% This is a demo for the LOMO feature extraction

clear;
clc;
close all;

%imgDir = 'PRID450S/cam_a/';
imgDir = 'ncamA/';
%addpath('../bin/');

%% Get image list
%list = dir(['PRID450S/cam_a/*.png']);
list = dir(['ncamA/*.png']);
n = length(list);

%% Allocate memory
%info = imfinfo([imgDir, list(1).name]);
%images = zeros(info.Height, info.Width, 3, n, 'uint8');

%% read images
for i = 1 : n
    info = imfinfo([imgDir, list(i).name]);
    images = zeros(info.Height, info.Width, 3,'uint8');
    images(:,:,:,i) = imread([imgDir, list(i).name]);
    images1(:,:,:,i) = imresize(images(:,:,:,i),[128 48]);
end

%% extract features. Run with a set of images is usually faster than that one by one, but requires more memory.
imshow(images1(:,:,:,i));
gallery = LOMO(images1);

%% if you need to set different parameters other than the defaults, set them accordingly
%{
options.numScales = 3;
options.blockSize = 10;
options.blockStep = 5;
options.hsvBins = [8,8,8];
options.tau = 0.3;
options.R = [3, 5];
options.numPoints = 4;

descriptors = LOMO(images, options);
%}

%rmpath('../bin/');
