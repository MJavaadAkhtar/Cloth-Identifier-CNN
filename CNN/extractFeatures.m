function extractFeatures(param)
images = dir('./images/*.jpg');
persons = dir('./labels/*person.png');
truths = load('fashionista_v0.2.1.mat');
cloths = dir('labels/*clothes.png');
labels = {};
% truths = truth.truths;
features = [];

%If the user doesnt tells which part it wants to solve, a or b
if nargin < 1
     error("Please choose the parameter to be either 'a' or 'b'");
end
%param decides whether we want to compute part a of the project or part b. 
for i = 1:6
    if param == 'a'
        backgroundAndPerson(i,persons,images,truths.truths,labels,features);
    elseif param == 'b'
        clothsDetector(i,cloths,images,truths.truths,labels,features);
    else
        error("Please choose the parameter to be either 'a' or 'b'");
    end
end
end