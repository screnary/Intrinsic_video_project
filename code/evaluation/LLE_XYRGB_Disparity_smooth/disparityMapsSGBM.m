% generate disparity map using SGBM method, for Temple; bad
root = './data/';
for i=1:40
%     leftimg  = [root 'GTimg/L' num2str(i,'%04d') '.png'];
%     rightimg = [root 'GTimg/R' num2str(i,'%04d') '.png'];
%     Il = rgb2gray(imread(leftimg));
%     Ir = rgb2gray(imread(rightimg));
%     d = disparity(Il, Ir, 'Method', 'SemiGlobal', 'BlockSize', 5);
%     imshow(d);
    Idis = imread([root, 'disparity/odp-nor-', num2str(i,'%04d'), '.png']);
    Idis_ref = rgb2gray(imread([root, 'GTimg/TL', num2str(i,'%04d'), '.png']));
    Idis_ref = padarray(Idis_ref,[0,7],61,'both');
%     Idis = padarray(Idis,[8,13],25,'both');
    
    Idis_1 = zeros(size(Idis));
    Idis_1(Idis~=25) = Idis(Idis~=25);
    Idis_ref_1 = zeros(size(Idis_ref));
    Idis_ref_1(Idis_ref~=61) = Idis_ref(Idis_ref~=61);
    Idis_1_new = uint8((double(Idis_1)-min(double(Idis_1(:))))/(max(double(Idis_1(:))) - min(double(Idis_1(:)))).*255);
    Idis_ref_1_new = uint8((double(Idis_ref_1)-min(double(Idis_ref_1(:))))/(max(double(Idis_ref_1(:))) - min(double(Idis_ref_1(:)))).*255);
    
%     Idis_new = uint8((double(Idis)-min(double(Idis(:))))/(max(double(Idis(:))) - min(double(Idis(:)))).*255);
%     Idis_ref_new = uint8((double(Idis_ref)-min(double(Idis_ref(:))))/(max(double(Idis_ref(:))) - min(double(Idis_ref(:)))).*255);
    imwrite(Idis_1_new, [root, 'disparity/rectified/', num2str(i,'%04d'), '.png']);
    imwrite(Idis_ref_1_new, [root, 'GTimg/rectified/TL', num2str(i,'%04d'), '.png']);
end
