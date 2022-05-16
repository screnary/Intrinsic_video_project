clear all;
totalMSEA = 0;
totalLMSEA = 0;
totalDSSIMA = 0;
totalMSES = 0;
totalLMSES = 0;
totalDSSIMS = 0;
count = 0;

DataSet = 'MIT';
result_folder = ['test-imgs_ep200', '/'];
inputDir = ['../',...
    '/ckpoints-Basic-MIT_scratch-self-sup+ms+fd+pers',...
    '-MIT-decoder_Residual/log/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-self-sup+ms+fd+pers-v5',...
%     '-MPI-main-RD-sceneSplit-decoder_Residual/log/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-v6_Twoway-ms+pers',...
%     '-MPI-main-RD-sceneSplit-decoder_Residual/log/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-v8-3_self-sup+ms',...
%     '-MPI-main-RD-sceneSplit-decoder_Residual/log/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-v8_new1_self-sup+ms+fd+pers',...
%     '-MPI-main-RD-sceneSplit-decoder_Residual/log/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-direct_intrinsics',...
%     '-MPI-main-RD-sceneSplit/log/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-v12-imagesplit_self-sup+ms+fd+pers',...
%     '-MPI-main-imageSplit-decoder_Residual/log/'];
DataDir = ['../datasets/',DataSet,'/'];
test_file = [DataDir, 'test.txt'];
% test_file = [DataDir, 'MPI_main_sceneSplit-fullsize-NoDefect-test.txt'];

% test_file = [DataDir, 'MPI_main_imageSplit-fullsize-ChenSplit-test.txt'];
images = importdata(test_file);  % a cell

disp([inputDir, result_folder]);

for n = 1:length(images)
    albedoName = [inputDir result_folder num2str(n-1) '_reflect-pred.png'];
    shadingName = [inputDir result_folder num2str(n-1) '_shading-pred.png'];
    labelAlbedoName = [inputDir result_folder num2str(n-1) '_reflect-real.png'];
    labelShadingName = [inputDir result_folder num2str(n-1) '_shading-real.png'];

    albedo = im2double(imread(albedoName));
    labelAlbedo = im2double(imread(labelAlbedoName));
    shading = im2double(imread(shadingName));
    labelShading = im2double(imread(labelShadingName));
    [height, width, channel] = size(albedo);

    totalMSEA = totalMSEA + evaluate_one_k(albedo,labelAlbedo);
    totalLMSEA = totalLMSEA + levaluate_one_k(albedo,labelAlbedo);
    totalDSSIMA = totalDSSIMA + (1-evaluate_ssim_one_k_fast(albedo,labelAlbedo))/2;

    totalMSES = totalMSES + evaluate_one_k(shading,labelShading);
    totalLMSES = totalLMSES + levaluate_one_k(shading,labelShading);
    totalDSSIMS = totalDSSIMS + (1-evaluate_ssim_one_k_fast(shading,labelShading))/2;

    count = count + 1;
    if length(images) >= 100 && mod(count,100)==0
        disp(count);
    elseif length(images) < 100
        disp(count);
    end
end
totalMSEA = totalMSEA/count;
totalLMSEA = totalLMSEA/count;
totalDSSIMA = totalDSSIMA/count;
totalMSES = totalMSES/count;
totalLMSES = totalLMSES/count;
totalDSSIMS = totalDSSIMS/count;
disp('albedo mse: shading mse:   albedo lmse: shading lmse:   albedo dssim: shading dssim:');
disp(sprintf('%f  %f\t%f  %f\t%f  %f', totalMSEA, totalMSES, totalLMSEA, totalLMSES, totalDSSIMA, totalDSSIMS));
% disp(sprintf('albedo: mse: %f, lmse: %f, dssim: %f',totalMSEA,totalLMSEA,totalDSSIMA));
% disp(sprintf('shading: mse: %f, lmse: %f, dssim: %f',totalMSES,totalLMSES,totalDSSIMS));