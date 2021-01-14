%% compute rmse|计算RMSE 
% 1`modify the directories 2`run|修改路径,再运行
clear;close all;clc

% mask directory|掩膜路径
maskdir = 'C:/资料/2019-08-去阴影/ISTD_Dataset/test/test_B/';
MD = dir([maskdir '/*.png']);

% result directory|结果路径
shadowdir = 'C:\Users\Administrator\Desktop\\model_istda_5\B_100\'; 
SD = dir([shadowdir '/*.png']);
% SD = dir([shadowdir '/*.jpg']);

% ground truth directory|GT路径_其他方法\test_C\'; 
freedir = 'C:\资料\2019-08-去阴影\AISTD\test_C_fixed_official/'; %AISTD
FD = dir([freedir '/*.png']);

% ours=1: compute RMSE (actually MAE) in shadow and non-shadow regions on each image first and then compute the average of all images
% ours=0: following the way of DSC or SP+M-Net to compute RMSE (actually MAE) 
ours=0;


total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
rl=zeros(1,size(SD,1)); 
ra=zeros(1,size(SD,1));
rb=zeros(1,size(SD,1));
nrl=zeros(1,size(SD,1));
nra=zeros(1,size(SD,1));
nrb=zeros(1,size(SD,1));
srl=zeros(1,size(SD,1));
sra=zeros(1,size(SD,1));
srb=zeros(1,size(SD,1));
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
% ISTD dataset image size 480*640
tic;
mask = ones([480,640]);
cform = makecform('srgb2lab');
for i=1:size(SD)
%     sname = strcat(shadowdir,'99-2.png'); 
    sname = strcat(shadowdir,SD(i).name); 
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);
%     s=imresize(s,[480 640]);
%     f=imresize(f,[480 640]);
%     m=imresize(m,[480 640]);
    mask = ones([size(f,1),size(f,2)]);

    nmask=~m;       %mask of non-shadow regions|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    
    f = double(f)/255;
    s = double(s)/255;
    
%     ppsnr(i)=psnr(s,f);
%     ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
%     ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
%     sssim(i)=ssim(s,f);
%     sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
%     sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    
    f = applycform(f,cform);
    s = applycform(s,cform);

    
    %abs lab
    absl=abs(f(:,:,1) - s(:,:,1));
    absa=abs(f(:,:,2) - s(:,:,2));
    absb=abs(f(:,:,3) - s(:,:,3));

    % rmse
    summask=sum(mask(:));
    rl(i)=sum(absl(:))/summask;
    ra(i)=sum(absa(:))/summask;
    rb(i)=sum(absb(:))/summask;
    
    if ours
        %% non-shadow, ours, per image
        distl = absl.* nmask;
        dista = absa.* nmask;
        distb = absb.* nmask;
        sumnmask=sum(nmask(:));
        nrl(i)=sum(distl(:))/sumnmask;
        nra(i)=sum(dista(:))/sumnmask;
        nrb(i)=sum(distb(:))/sumnmask;

        % shadow
        distl = absl.* smask;
        dista = absa.* smask;
        distb = absb.* smask;
        sumsmask=sum(smask(:));
        srl(i)=sum(distl(:))/sumsmask;
        sra(i)=sum(dista(:))/sumsmask;
        srb(i)=sum(distb(:))/sumsmask;
        disp(i);
    else
        %% rmse in shadow, original way, per pixel
        dist = abs((f - s).* repmat(smask,[1 1 3]));
        total_dists = total_dists + sum(dist(:));
        total_pixels = total_pixels + sum(smask(:));
        % rmse in non-shadow, original way, per pixel
        dist = abs((f - s).* repmat(nmask,[1 1 3]));
        total_distn = total_distn + sum(dist(:));
        total_pixeln = total_pixeln + sum(nmask(:));  
        disp(i);
    end
end
toc;
if ours
    %% non-shadow, ours, per image
%     fprintf('L-channel(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(rl),mean(nrl),mean(srl));
%     fprintf('a-channel(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(ra),mean(nra),mean(sra));
%     fprintf('b-channel(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(rb),mean(nrb),mean(srb));
    fprintf('Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(rl)+mean(ra)+mean(rb),mean(nrl)+mean(nra)+mean(nrb),mean(srl)+mean(sra)+mean(srb));
%     fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
%     fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
else
    %% rmse in shadow, original way, per pixel
    fprintf('all,non-shadow,shadow:\n%f\t%f\t%f\n\n',mean(rl)+mean(ra)+mean(rb),total_distn/total_pixeln,total_dists/total_pixels);
end
    
