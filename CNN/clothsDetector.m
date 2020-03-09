function clothsDetector(i,cloths,images,truth,labels,features)
    % This function computes the different kind of clothings and store the labels
    % and featuers for further use for CNN. i is the number of the current
    % image, persons is the directory of labels/*person.jpg, truths is the
    % truth extracted from fashionista_v0.2.1, label_image is where the
    % labels of the final computed image is going to be stored and features
    % is where the features of the function is going to be stored.

    current = imread(images(i).name);
    truths = truth(i).pose.point;
    labelimage = imread(cloths(i).name);
    A = current;
    A = imresize(A,0.5);
    Alab = rgb2lab(A);
    %Computing the super pixels and Convert label matrix to cell array of
    %linear indices which is obtained by the super pixels
    [SP, N] = superpixels(current,200,'isInputLab',true);
    [rSp,~] = size(SP);
    SP_index = label2idx(SP);
    counter = length(SP_index);
    BW = boundarymask(SP);
%     imshow(imoverlay(current,BW,'cyan'),'InitialMagnification',67);
%     hold on;
    
    for j = 1:counter
        
        %We are finding the center of the each segment of the super pixels
        %for labeling purposes.
        diffX = max(mod(SP_index{j}, rSp)) - min(mod(SP_index{j}, rSp));
        diffY = (max(SP_index{j}/rSp) + 1) - (min(SP_index{j}/rSp) + 1);
        centerX = ceil(min(mod(SP_index{j}, rSp)) + diffX/2);
        centerY = ceil((min(SP_index{j}/rSp) + 1) + diffY/2);
        
        % Plotting the points on the image.
%         plot(centerY,centerX,'m.','MarkerSize',10);
        
        %Filtering out the super pixels which are below the threshold as
        %described in the report. 
        if centerX-11 <= 0 || centerX+12 > 600 || centerY-12 <= 0 || centerY+11 > 400
             continue;
        end
        
        % Extracting the HOG features from the image, so it can helpfull for
        % fruther use when using the CNN
        [featuresExtract, ~] = extractHOGFeatures(current, [centerX centerY], 'CellSize', [8 8]);
        if isempty(featuresExtract) == 0
            %labelling each super pixel segmentation with 0 if it is a
            %background and other labels if it is a differnt piece of clothings from the knowledge of
            %the truth image from the label
%             text(centerY,centerX,int2str(labelimage(centerX,centerY)),'Color','white','FontSize',14);
            distances = zeros(1,14);
            for m = 1:14
                distances(m) = sqrt( (centerX - truths(m,1))^2  + (centerY - truths(m,2))^2 );
            end
            features = [features; featuresExtract];
            labels = [labels; labelimage(centerX,centerY)];
        end
    end
    
    %coloring the region of the super pixels so we can distinguigh certain
    %elements
    meanColor = zeros(N,3);
    [m,n] = size(SP);
    for  i = 1:N
            meanColor(i,1) = mean(Alab(SP_index{i}));
            meanColor(i,2) = mean(Alab(SP_index{i}+m*n));
            meanColor(i,3) = mean(Alab(SP_index{i}+2*m*n));
    end
    numColors = 4;
    [idx,cmap] = kmeans(meanColor,numColors,'replicates',2);
    cmap = lab2rgb(cmap);
    Lout = zeros(size(A,1),size(A,2));
    for i = 1:N
        Lout(SP_index{i}) = idx(i);
    end
%     imshow(label2rgb(Lout))
    imshow(Lout,cmap)

end