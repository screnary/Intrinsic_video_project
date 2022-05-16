data_root = '../ckpoints--iiw_v7-oneway+pixsupv2+vgg19-IIW-decoder_Residual/log/';
% data_dir = [data_root, 'test-imgs_ep12-renamed/'];
data_dir = [data_root, 'test-imgs_ep12/'];
out_dir = [data_root, 'test-imgs_ep12-scaled/'];

ref_dir = '../IntrinsicImage-master/results/IIW_combine-with_rescale/';
file_list = importdata(['../datasets/IIW/', 'test_list.txt']);

% change tune
total_mu_r = [];
total_mu_r_ref = [];
total_std_r = [];
total_std_r_ref = [];

total_mu_s = [];
total_mu_s_ref = [];
total_std_s = [];
total_std_s_ref = [];
for i = 1:length(file_list)
    id = file_list{i}(1:end-4);
    ref_i = im2double(imread([ref_dir, id, '.png']));
    ref_r = im2double(imread([ref_dir, id, '-R.png']));
    ref_s = im2double(imread([ref_dir, id, '-S.png']));
    
%     img_i = im2double(imread([data_dir, id, '_input.png']));
%     img_r = im2double(imread([data_dir, id, '_reflect-pred.png']));
%     img_s = im2double(imread([data_dir, id, '_shading-rec.png']));
    
    img_i = im2double(imread([data_dir, id, '.png']));
    img_r = im2double(imread([data_dir, id, '_r.png']));
    img_s = im2double(imread([data_dir, id, '_sr.png']));
    
    [h, w, ~] = size(img_i);
    ref_i = imresize(ref_i, [h,w]);
    ref_r = imresize(ref_r, [h,w]);
    ref_s = imresize(ref_s, [h,w]);
    
    lab_ref_r = rgb2lab(ref_r);
    lab_img_r = rgb2lab(img_r);
    lab_img_i = rgb2lab(img_i);
    
    L_ref_r = lab_ref_r(:,:,1);
    L_img_r = lab_img_r(:,:,1);
    L_img_i = lab_img_i(:,:,1);
    
    L_ref_s = 100 * ref_s;
    L_img_s = 100 * img_s(:,:,1);
    
    mu_ref = mean(L_ref_r(:));
    std_ref = std(L_ref_r(:));
    
    mu_r = mean(L_img_r(:));
    std_r = std(L_img_r(:));
    
    mu_i = mean(L_img_i(:));
    std_i = std(L_img_i(:));
    
    mu_ref_s = mean(L_ref_s(:));
    std_ref_s = std(L_ref_s(:));
    
    mu_s = mean(L_img_s(:));
    std_s = std(L_img_s(:));
    
    total_mu_r = [total_mu_r; mu_r];
    total_mu_r_ref = [total_mu_r_ref; mu_ref];
    total_std_r = [total_std_r; std_r];
    total_std_r_ref = [total_std_r_ref; std_ref];

    total_mu_s = [total_mu_s; mu_s];
    total_mu_s_ref = [total_mu_s_ref; mu_ref_s];
    total_std_s = [total_std_s; std_s];
    total_std_s_ref = [total_std_s_ref; std_ref_s];
    
    %% shift distribution of reflectance
%     if 1.35*std_ref < std_r
%         scale_factor = 1.0;
%     else
%         scale_factor = 1.35;
%     end
%     L_r_shift = (L_img_r - mu_r) * scale_factor*std_ref / std_r + 1.0*mu_ref;
%     
%     
%     lab_new_r = lab_ref_r;
%     lab_new_r(:,:,1) = L_r_shift;   % direct change is bad
%     new_img_r = lab2rgb(lab_new_r);
    
    %% shift distribution of shading
    L_s_shift = (L_img_s - mu_s) * 1.0*std_ref_s / std_s + 1.0*mu_ref_s;
    if max(L_s_shift(:) > 100)
        L_s_shift = L_s_shift / max(L_s_shift(:));
    else
        L_s_shift = L_s_shift / 100.0;
    end

    new_img_s = repmat(L_s_shift, [1,1,3]);
%     imwrite(new_img_s, [data_dir, id, '_s_scaled.png']);
    
    %% save
%     imwrite(new_img_r, [out_dir, id, '_r.png']);
%     imwrite(img_i, [out_dir, id, '.png']);
%     imwrite(img_s, [out_dir, id, '_sr.png']);

%     lab_new_r_1 = lab_img_r;
%     lab_new_r_1(:,:,1) = L_r_shift;   % direct change is bad
%     new_img_r_1 = lab2rgb(lab_new_r_1);
%     
%     figure, imshow([ref_r, new_img_r, new_img_r_1, img_r]);
    
end