function [x,y] = Q1a(image)

img  = rgb2gray(image);

%computing the gradient and (I_x)^2 and (I_Y)^2 and I_xy
[GX,GY] = imgradientxy(img);
IX2 = GX.^2;
IY2 = GY.^2;
IXY = GX.*GY;

% Computing M
Ix2 = conv2(IX2, fspecial('gaussian'));
Iy2 = conv2(IY2, fspecial('gaussian'));
Ixy = conv2(IXY, fspecial('gaussian'));
[h,w] = size(img);

% Computing R = det(M) - alpha*trace(M)^2
determinant = Ix2.*Iy2 - Ixy.^2;
Trace = Ix2 + Iy2;
R = determinant - (0.05).*((Trace).^2);
maximum = max(max(R));

%Now we will try to do the non maximum surppression
new_matrix = zeros(h,w);
for i = 1:h
    for j = 1:w
        %(maximum)*0.028 is the threshhold we placed and the surroundings
        if R(i,j) > (maximum)*0.035
            local_max = true;
            for m = i-1:i+1
                for n = j-1:j+1
                    if R(m,n) > R(i,j)
                        local_max = false;
                        break
                    end
                end
                if local_max == false
                    break
                end
                if local_max == true
                    new_matrix(i,j) = 1;
                end
            end
        end
    end
end

[x,y] = find(new_matrix == 1);
imshow(image);
hold on;
plot(y,x,'r.');
end