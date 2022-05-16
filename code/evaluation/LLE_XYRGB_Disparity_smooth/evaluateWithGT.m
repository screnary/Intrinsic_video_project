% evaluate disparity result with Ground Truth
clear,
close all,
clc,
root = './data/';
demo = 'temple';
SaveFlag = 0;


err_dis_vec = [];
err_dis_lle_vec = [];
err_dis_smo_vec = [];
for num = 1:40
    % %read data
    if strcmp(demo, 'temple')
        Idis = imread([root, 'disparity/rectified/', num2str(num,'%04d'), '.png']);
        gtdis= imread([root, 'GTimg/rectified/TL',   num2str(num,'%04d'), '.png']);
        Idis_lle = imread([root, demo, '/output/res-', num2str(num), '.png']);
        Idis_smooth = imread([root, demo, '/videofilter/', num2str(num,'%04d'), '.png']);
    elseif strcmp(demo, 'book')
        Idis  = rgb2gray(imread([root, demo, '/disparity/compare/cut/cut2/frame_', num2str(num,'%04d'), '.png']));
        gtdis = rgb2gray(imread([root, demo, '/TL',   num2str(num,'%04d'), '.png']));
        gtdis = padarray(gtdis, [0,1], 'replicate', 'both');
        Idis_lle = imread([root, demo, '/output/res-', num2str(num), '.png']);
        Idis_smooth = imread([root, demo, '/videofilter/', num2str(num,'%04d'), '.png']);
    end
    % %compute difference with gt
    
    tol = 10;
    Error_dis        = abs(double(Idis)        - double(gtdis)) > tol;
    Error_dis_lle    = abs(double(Idis_lle)    - double(gtdis)) > tol;
    Error_dis_smooth = abs(double(Idis_smooth) - double(gtdis)) > tol;
    
    Error_dis_value        = uint8(abs(double(Idis)        - double(gtdis)));
    Error_dis_lle_value    = uint8(abs(double(Idis_lle)    - double(gtdis)));
    Error_dis_smooth_value = uint8(abs(double(Idis_smooth) - double(gtdis)));
    
    err_dis     = sum(Error_dis(:));
    err_dis_lle = sum(Error_dis_lle(:));
    err_dis_smo = sum(Error_dis_smooth(:));
    baseNum     = size(Error_dis,1)*size(Error_dis,2);
    err_dis_vec = [err_dis_vec, err_dis];
    err_dis_lle_vec = [err_dis_lle_vec, err_dis_lle];
    err_dis_smo_vec = [err_dis_smo_vec, err_dis_smo];
    
    if SaveFlag == 1;
        savedir = [root, demo, '/evaluate/tol=', num2str(tol)];
        if ~isdir(savedir)
            mkdir(savedir);
        end
    figure,
    subplot(2,3,1), imshow(Error_dis_value), title('Error_dis_value');
    subplot(2,3,2), imshow(Error_dis_lle_value), title('Error_dis_lle_value');
    subplot(2,3,3), imshow(Error_dis_smooth_value), title('Error_dis_smooth_value');
    
    subplot(2,3,4), imshow(Error_dis), title('Error_dis');
    subplot(2,3,5), imshow(Error_dis_lle), title('Error_dis_lle');
    subplot(2,3,6), imshow(Error_dis_smooth), title('Error_dis_smooth');
    
    saveas(gcf, [savedir '/show-', num2str(num), '.png']);
    if mod(num,10) == 0
        close all;
    end
    end
end

acc_vector = [err_dis_vec;err_dis_smo_vec;err_dis_lle_vec]./baseNum;
figure, bar(acc_vector);
saveas(gcf, ['./data/' demo '/evaluate/bar-tol_', num2str(tol), '' '.png']);
mean(acc_vector,2)
