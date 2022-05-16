% image RGB xy space guided disparity smoothing(filling hole), Using LLE
% input data
clear,
close all,
clc,
root = './data/';
demo = 'book'; %change this, beetle
for num=1:41
    num
    if strcmp(demo, 'temple') % num=1:30
        Idis = imread([root, 'disparity/rectified/', num2str(num,'%04d'), '.png']);
        gtdis= imread([root, 'GTimg/rectified/TL',   num2str(num,'%04d'), '.png']);
        Irgb = imread([root, 'GTimg/L',            num2str(num,'%04d'), '.png']);
        Irgb = padarray(Irgb,[0,7,0],'replicate','both');
        [h,w,c] = size(Irgb);
        % rgb feature space
        RGB = reshape(Irgb, [h*w,3]);
        
        % Lab feature space
        Ilab = rgb2lab(Irgb);
        Lab = reshape(Ilab, [h*w,3]);
%         Idis = Idis(:,8:367);
        if size(gtdis,3)>1
            Idis_ref = rgb2gray(gtdis);
        else
            Idis_ref = gtdis;
        end
        
%         Idis = padarray(Idis,[0,7],61,'both'); % the size of Idis is not right
    elseif strcmp(demo, 'beetle')
        I_rgb = imresize(imread([root, demo, '/src/left-', num2str(130+num,'%04d'), '.jpg']), 1);
        I_dis = imresize(imread([root, demo, '/disparity/source-disp-', num2str(num,'%04d'), '.jpg']),1);
        
        rect = [350,370,170,190];
        Irgb = imcrop(I_rgb, rect);
        Idis = imcrop(I_dis, rect);
        [h,w,c] = size(Irgb);
        RGB = reshape(Irgb, [h*w,3]);
        Ilab = rgb2lab(Irgb);
        Lab = reshape(Ilab, [h*w,3]);
    elseif strcmp(demo, 'book')
%         I_dis_ref = rgb2gray(imresize(imread([root, demo, '/disparity/compare/cut/cut2/frame_', num2str(num-1,'%04d'), '.png']),1));
        I_dis_ref = rgb2gray(imresize(imread([root, demo, '/TL', num2str(num,'%04d'), '.png']),1));
        I_dis_ref = padarray(I_dis_ref, [0,1], 'replicate', 'both');
        I_rgb = imresize(imread([root, demo, '/cut1-yuantu/frame_', num2str(num,'%04d'), '.png']), 1);
        I_dis = imresize(imread([root, demo, '/disparity/compare/cut/cut2/frame_', num2str(num,'%04d'), '.png']),1);
        I_dis = rgb2gray(I_dis);
%         rect = [40,1,270,195];
        rect = [1,1,size(I_dis,2),size(I_dis,1)];
        
        Idis_ref = imcrop(I_dis_ref, rect);
        Irgb = imcrop(I_rgb, rect);
        Idis = imcrop(I_dis, rect);
        [h,w,c] = size(Irgb);
        RGB = reshape(Irgb, [h*w,3]);
        Ilab = rgb2lab(Irgb);
        Lab = reshape(Ilab, [h*w,3]);
    end

    % feature distance and k nearest neighbors
    [I,J] = ind2sub([h,w],1:h*w);
    X = [I;J;Lab']; % data D*N

    %% find the points need to be computed (for inpainting)
    K = 12; %12
    fprintf(1,'-->Finding %d nearest neighbours.\n',K);
    if strcmp(demo, 'temple') 
%         holeIndex = find(Idis ~= 25 & Idis < 115 | Idis == 217); % the black holes points, 75
        Index = Idis_ref>Idis+25 | Idis_ref<Idis-25;
        holeIndex = find(Index);
        figure, imshow(Index);
    elseif strcmp(demo, 'beetle')
        holeIndex = find(Idis < 115);
    elseif strcmp(demo, 'book')
%         MASK = zeros(size(Idis));
%         MASK(1:80,100:220) = 1;  % when num == 25
%         holeIndex = find(MASK);
        
        Index = Idis_ref>Idis+15 | Idis_ref<Idis-5;
        holeIndex = find(Index);
%         figure, imshow(Index);
    end
    N = length(holeIndex)
    D = size(X,1);
    Nei = zeros(N,K);
    X_hole = X(:, holeIndex); 
    for i = 1:size(X_hole,2)
        X_tmp = repmat(X_hole(:,i),1,size(X,2));
        distance = sum((X-X_tmp).^2,1); % distance vector
        [B, I] = sort(distance); % B=distance(I)
        neighbor_index = I(~ismember(I,holeIndex));
%         neighbor_index = I(ismember(I,holeIndex));
%         neighbor_index = I(:);
        Nei(i,:) = neighbor_index(1:K);
    end

    %% construct reconstruction weights
    fprintf(1,'-->Solving for reconstruction weights.\n');
    if(K>D) 
      fprintf(1,'   [note: K>D; regularization will be used]\n'); 
      tol=1e-3; % regularlizer in case constrained fits are ill conditioned
    else
      tol=0;
    end

    W = zeros(K, N);
    for ii = 1:N
        z = X(:, Nei(ii,:)) - repmat(X(:,ii),1,K); % shift ith pt to origin
        C = z'*z;                                  % local covariance
        C = C + diag(diag(C));
        C = C + eye(K,K)*tol*trace(C);             % regularlization (K>D)
        W(:,ii) = C\ones(K,1);                     % solve Cw=1
        W(:,ii) = W(:,ii)/sum(W(:,ii));            % enforce sum(w)=1
    end

    % interpolate implanting values
    % because the neighbors are outside from the set, we can compute directly
    dis_new = zeros(N,1);
    for jj = 1:N
        dis_new(jj) = double(Idis(Nei(jj,:))) * W(:,jj);
    end
    Idis_new = Idis;
    Idis_new(holeIndex) = dis_new;
%     if strcmp(demo, 'book')
%         I_dis_new = I_dis;
%         I_dis_new(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = Idis_new;
%         figure,imshow(I_dis_new)
%     end
    figure('Name','completion result');
    subplot(2,2,1);
    imshow(Idis);
    title('before');
    subplot(2,2,2);
    imshow(Idis_new);
    title('after');
    subplot(2,2,3);
    imshow(Irgb);
    title('rgb');
    subplot(2,2,4);
    imshow(Index);
    title('hole pixes');
    saveas(gcf, ['./data/' demo '/output/show-', num2str(num), '.png']);
    imwrite(Idis_new, ['./data/' demo '/output/res-', num2str(num), '.png'])
    if mod(num,5)==0
        close all;
    end
    % filtering optimization

end