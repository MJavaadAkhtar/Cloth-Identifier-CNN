%VOCEVALSEG Evaluates a set of segmentation results.
% VOCEVALSEG(VOCopts,ID); prints out the per class and overall
% segmentation accuracies. Accuracies are given using the intersection/union 
% metric:
%   true positives / (true positives + false positives + false negatives) 
%
% [ACCURACIES,AVACC,CONF] = VOCEVALSEG(VOCopts,ID) returns the per class
% percentage ACCURACIES, the average accuracy AVACC and the confusion
% matrix CONF.
%
% [ACCURACIES,AVACC,CONF,RAWCOUNTS] = VOCEVALSEG(VOCopts,ID) also returns
% the unnormalised confusion matrix, which contains raw pixel counts.
function [accuracies,avacc,conf,rawcounts] = evalseg(resdir, gtids, whicheval)

% whicheval=1  to evaluate person
% whicheval=2 to evaluate clothing
globals;

gt_dir = fullfile(DATA_DIR, 'labels');
% evaluating assign

if nargin < 1
    %resdir = fullfile(DATA_DIR, 'results-seg');
    resdir = gt_dir;
end;
if nargin < 2;
   gt_filename = fullfile(DATA_DIR, [imset '.txt']);
   fid = fopen(gt_filename, 'r+');
   gtids = textscan(fid, '%s');
   gtids = gtids{1};
   fclose(fid);
end;


if whicheval==1
   classes{1} = 'person';
   labext = '_person';
else
   classes{1} = 'skin';
   classes{2} = 'hair';
   classes{3} = 'tshirt';
   classes{4} = 'shoes';
   classes{5} = 'pants';
   classes{6} = 'dress';
   labext = '_clothes';
end;

% number of labels = number of classes plus one for the background
num = length(classes) + 1; 
nclasses = length(classes);
confcounts = zeros(num);
count=0;
tic;
fprintf('testing %d images\n', length(gtids))
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('confusion: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
        
    imname = gtids{i};
    
    % ground truth label file
    gtfile = fullfile(gt_dir, [imname labext '.png']);
    if ~exist(gtfile, 'file')
        fprintf('gt file doesnt exist, skipping\n');
        continue;
    end;
    [gtim,map] = imread(gtfile);    
    gtim = double(gtim);
    
    % results file
    resfile = fullfile(resdir, [imname labext '.png']);
    if exist(resfile, 'file')
       [resim,map] = imread(resfile);
    else
        resim = zeros(size(gtim));
    end;
    resim = double(resim);

    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim~=szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end
    
    %pixel locations to include in computation
    locs = gtim<255;
    gtim(locs) = double(gtim(locs));
    
    % joint histogram
    sumim = 1+gtim+resim*num; 
    hs = histc(sumim(locs),1:num*num); 
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);
end

% confusion matrix - first index is true label, second is inferred label
%conf = zeros(num);
conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
rawcounts = confcounts;

% Percentage correct labels measure is no longer being used.  Uncomment if
% you wish to see it anyway
%overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
%fprintf('Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

accuracies = zeros(nclasses,1);
fprintf('Accuracy for each class (intersection/union measure)\n');
for j=1:num
   
   gtj=sum(confcounts(j,:));
   resj=sum(confcounts(:,j));
   gtjresj=confcounts(j,j);
   % The accuracy is: true positive / (true positive + false positive + false negative) 
   % which is equivalent to the following percentage:
   accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);   
   
   clname = 'background';
   if (j>1), clname = classes{j-1};end;
   fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
end
accuracies = accuracies(2:end);
avacc = mean(accuracies);
fprintf('-------------------------\n');
for i=1:length(classes)
   fprintf('%s accuracy: %6.3f%%\n',classes{i}, avacc);
end;
