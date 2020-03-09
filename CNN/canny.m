function canny()
addpath(genpath('./images'));
addpath(genpath('./labels'));
images = dir('images/*.jpg');


for all= 1:5
    img =  rgb2gray(double(imread(images(all).name))/255) ;
    % img =  double(imread('002.png'))/255 ;
    f1 = edge(img, 'canny', 0.3, 0.3);
    [x,y] = Q1a(imread(images(all).name));
    figure;
    imshow(f1);
    [l,w] = size(f1);
    count=0;
    for i = 1:w
        for j = 1:l-1
            count = count + 1;
            if f1(j,i) == 1 
                f1(j+3,i) = 1;         
            end
        end
    end
     imwrite(f1,strcat('canny',images(all).name), 'jpg');
end

end