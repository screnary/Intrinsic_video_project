clear all;
% DataSet = 'MIT';
% % result_folder = ['FD+pers-fromMPI-test-imgs_ep190', '/'];  % not good
% result_folder = ['FD+pers-test-imgs_ep120', '/'];
% inputDir = ['../result-data/',DataSet,'/'];
% % inputDir = ['../result-data/',DataSet,'/'];
% DataDir = ['../datasets/',DataSet,'/'];

DataSet = 'MIT';
% result_folder = ['test-imgs_ep450', '/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-MIT_v1_continue-self-sup+ms+fd+pers',...
%     '-MIT-decoder_Residual/log/'];
% out_folder = ['test-imgs_ep450-masked', '/'];

% result_folder = ['test-imgs_ep200', '/'];
% inputDir = ['../',...
%     '/ckpoints-Basic-MIT_scratch-self-50sup+ms+fd+pers',...
%     '-MIT-decoder_Residual/log/'];
% out_folder = ['test-imgs_ep200-masked', '/'];

% DataDir = ['../datasets/',DataSet,'/'];
% test_file = [DataDir, 'test.txt'];
% images = importdata(test_file);  % a cell

% for m =1:length(images)
%     disp(images{m})
%     maskname_label = [DataDir, 'MIT-mask-fullsize/', images{m}];
%     albedoname_predict = [inputDir result_folder num2str(m-1) '_reflect-pred.png'];
%     shadingname_predict = [inputDir result_folder num2str(m-1) '_shading-pred.png'];
%     albedoname_label = [inputDir result_folder num2str(m-1) '_reflect-real.png'];
%     shadingname_label = [inputDir result_folder num2str(m-1) '_shading-real.png'];
%     
%     albedo_predict = im2double(imread(albedoname_predict));
%     shading_predict = im2double(imread(shadingname_predict));
%     albedo_label = im2double(imread(albedoname_label));
%     shading_label = im2double(imread(shadingname_label));
%     mask = (imread(maskname_label));
%     V = mask > 0;
% 
%     V3 = repmat(V,[1,1,size(shading_label,3)]);
%     
%     albedo_p = albedo_predict .* V3;
%     shading_p = shading_predict .* V3;
%     imwrite(albedo_p, [outputDir, num2str(m-1), '_reflect-pred.png']);
%     imwrite(shading_p, [outputDir, num2str(m-1), '_shading-pred.png']);
%     imwrite(albedo_label, [outputDir, num2str(m-1), '_reflect-real.png']);
%     imwrite(shading_label, [outputDir, num2str(m-1), '_shading-real.png']);
% end
%% Revisiting, Fan et al. 2018
% result_folder = ['MIT-input-fullsize', '/'];
% inputDir = ['../',...
%     '/IntrinsicImage-master/',...
%     'results/MIT/'];
% out_folder = ['MIT-input-fullsize-masked', '/'];
% 
% outputDir = [inputDir, out_folder];
% if ~isdir(outputDir)
%     mkdir(outputDir);
% end
% 
% DataDir = ['../datasets/',DataSet,'/'];
% test_file = [DataDir, 'test.txt'];
% images = importdata(test_file);  % a cell
% 
% for m =1:length(images)
%     imname = images{m}(1:end-4);
%     disp(imname);
%     maskname_label = [DataDir, 'MIT-mask-fullsize/', images{m}];
%     albedoname_predict = [inputDir result_folder imname '-predict-albedo.png'];
%     shadingname_predict = [inputDir result_folder imname '-predict-shading.png'];
%     albedoname_label = [inputDir result_folder imname '-label-albedo.png'];
%     shadingname_label = [inputDir result_folder imname '-label-shading.png'];
%     
%     albedo_predict = im2double(imread(albedoname_predict));
%     shading_predict = im2double(imread(shadingname_predict));
%     albedo_label = im2double(imread(albedoname_label));
%     shading_label = im2double(imread(shadingname_label));
%     mask = (imread(maskname_label));
%     V = mask > 0;
% 
%     V3 = repmat(V,[1,1,size(shading_label,3)]);
%     
%     albedo_p = albedo_predict .* V3;
%     shading_p = shading_predict .* V3;
%     imwrite(albedo_p, [outputDir, num2str(m-1), '_reflect-pred.png']);
%     imwrite(shading_p, [outputDir, num2str(m-1), '_shading-pred.png']);
%     imwrite(albedo_label, [outputDir, num2str(m-1), '_reflect-real.png']);
%     imwrite(shading_label, [outputDir, num2str(m-1), '_shading-real.png']);
% end

%% direct intrinsics, 2014, MSCR
result_folder = ['di_mit', '/'];
inputDir = ['../',...
    '/direct_intrinsics/',...
    'results/'];
out_folder = ['di_mit-masked', '/'];

outputDir = [inputDir, out_folder];
if ~isdir(outputDir)
    mkdir(outputDir);
end

DataDir = ['../datasets/',DataSet,'/'];

images = {'cup2-1.png', 'deer-1.png', 'frog2-1.png', 'paper2-1.png', 'pear-1.png',...
    'potato-1.png', 'raccoon-1.png', 'sun-1.png', 'teabag1-1.png', 'turtle-1.png'};

for m =1:length(images)
    imname = images{m}(1:end-4);
    disp(imname);
    maskname_label = [inputDir, '../data/', images{m}(1:end-6), '/mask.png'];
    albedoname_predict = [inputDir result_folder num2str(m-1) '_reflect-pred.png'];
    shadingname_predict = [inputDir result_folder num2str(m-1) '_shading-pred.png'];
    
    albedo_predict = im2double(imread(albedoname_predict));
    shading_predict = im2double(imread(shadingname_predict));
    mask = (imread(maskname_label));
    V = mask > 0;

    V3 = repmat(V,[1,1,size(shading_predict,3)]);
    
    albedo_p = albedo_predict .* V3;
    shading_p = shading_predict .* V3;
    imwrite(albedo_p, [outputDir, imname, '_reflect-pred.png']);
    imwrite(shading_p, [outputDir, imname, '_shading-pred.png']);
end
