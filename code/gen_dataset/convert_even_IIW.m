dataroot = '../datasets/IIW/';
inputDir = [dataroot, 'data/'];
outputDir = [dataroot, 'data_even/'];
files = importdata([dataroot, 'iiw_Learning_Lightness_test.txt']);
% files = dir([inputDir '*.png']);
for x = 1:length(files)
	filename = [inputDir files{x}];
	outfile = [outputDir files{x}];

	img_in = imread(filename);
    [h,w,ch] = size(img_in);
    if mod(h, 2)
        h_o = h + 1;
    else
        h_o = h;
    end
    if mod(w, 2)
        w_o = w + 1;
    else
        w_o = w;
    end
    img_out = zeros(h_o, w_o, ch, 'uint8');
    img_out(1:h, 1:w, :) = img_in;
    
    imwrite(img_out, outfile);
end

