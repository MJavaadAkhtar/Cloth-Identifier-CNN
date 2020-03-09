function  backgroundAndPerson(i,persons,images,truth,label_image,features)
    % This function computes the background and person and store the labels
    % and featuers for further use for CNN. i is the number of the current
    % image, persons is the directory of labels/*person.jpg, truths is the
    % truth extracted from fashionista_v0.2.1, label_image is where the
    % labels of the final computed image is going to be stored and features
    % is where the features of the function is going to be stored.
    current = imread(images(i).name);
    truths = truth(i).pose.point;
    labelimage = imread(persons(i).name);
    
    %Computing the super pixels and Convert label matrix to cell array of
    %linear indices which is obtained by the super pixels
    [SP, ~] = superpixels(current,170);
    [rSp,~] = size(SP);
    SP_index = label2idx(SP);
    counter = length(SP_index);
    BW = boundarymask(SP);
    %Showing the image
    imshow(imoverlay(current,BW,'cyan'),'InitialMagnification',67);
    hold on;
    
    for j = 1:counter
        
        %We are finding the center of the each segment of the super pixels
        %for labeling purposes.
        diffX = max(mod(SP_index{j}, rSp)) - min(mod(SP_index{j}, rSp));
        diffY = (max(SP_index{j}/rSp) + 1) - (min(SP_index{j}/rSp) + 1);
        centerX = ceil(min(mod(SP_index{j}, rSp)) + diffX/2);
        centerY = ceil((min(SP_index{j}/rSp) + 1) + diffY/2);
 
        % Plotting the points on the image.
        plot(centerY,centerX,'m.','MarkerSize',10);
        
         %Filtering out the super pixels which are below the threshold as
         %described in the report. 
        if centerX-11 <= 0 || centerX+12 > 600 || centerY-12 <= 0 || centerY+11 > 400
             continue;
        end
        
        % Extracting the features from the image, so it can helpfull for
        % fruther use when using the CNN
        featuresExtract = extractLBPFeatures(rgb2gray(current(centerX-11: centerX+12, centerY-12: centerY+11, 1:3)));
%         [featuresExtract, ~] = extractHOGFeatures(current, [centerX centerY], 'CellSize', [8 8]);
        if isempty(featuresExtract) == 0
            
            %labelling each super pixel segmentation with 0 if it is a
            %background and 1 if it is a forground from the knowledge of
            %the truth image from the label
            text(centerY,centerX,int2str(labelimage(centerX,centerY)));
            features = [features; featuresExtract];
            label_image = [label_image; labelimage(centerX,centerY)];
        end
    end
%     % Person and background distinguisher
%     labels = [];
%     points=[[1,2];[2,3];[3,4];[4,5];[5,6];[7,8];[8,9];[9,10];[10,11];[11,12];[13,14]; [3,9]; [3,13]; [3,10]; [4,9]; [4,13]; [4,10];[10,14];[9,14]];
%     tempX= [];
%     tempY = [];
%     
%     for x = 1:length(points)
%         [x_p,y_p]=findLines([truths(points(x,1),1),truths(points(x,1),2)],[truths(points(x,2),1),truths(points(x,2),2)],100);
%         tempX = [tempX ,round(x_p)];
%         tempY = [tempY , round(y_p)];
%     end
%     
%     for x = 1:length(tempX)
%         if ~isnan(tempY(x)) && ~isnan(tempX(x))
%            labels = [labels SP(tempY(x), tempX(x))];
%         end
%     end
%     labels = unique(labels);
%     outputImage = zeros(600, 400);
%     
%     display(labels)
%     % labels all the super pixels that crosses the lines
%     for y = 1:600
%         for x = 1 : 400
%             if any( labels == SP(y,x))
%                 outputImage(y,x)= 1;
%             end
%         end
%     end
%     imagesc(outputImage);
%     imwrite(outputImage,images(i).name, 'jpg');
%     display(labels)
    
end