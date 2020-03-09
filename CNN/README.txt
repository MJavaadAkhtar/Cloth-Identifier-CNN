This folder contains data for Project 3 for the CSC420 class: http://www.cs.utoronto.ca/~fidler/teaching/2015/CSC420.html

The data is adapted from the following paper:

Kota Yamaguchi, M Hadi Kiapour, Luis E Ortiz, Tamara L Berg, "Parsing Clothing in Fashion Photographs", CVPR 2012
http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/

Explanation of the data:

- images/*.jpg    600 images
- labels/*_person.png   contains image labeling into person and background. If pixel has value 1, it belongs to the person class, otherwise it is background
- labels/*_clothes.png  contain image labeling for 6 clothing types and background. See labels.txt for the label information.

Function evalseg evaluates performance (please check the code how to run it), you will also need to set the data directory in 'globals.m'

TRAIN/TEST split:
- Suggested split is 50% for training, 10% validation, 40% for testing. So you could just take images 1-300 for training, 301-360 for validation and the rest for test
- If the above is too much data to process you can also make your own split. If you do not use any training, then you can test on all images, for example.

Some possible ideas how to attack the problem:
1. Compute superpixels for each image. Define some features on each superpixel, and train a classifier
2. Detect the person (with the 'person_final' model from Assignment 4), perform classification only inside the box.
3. Compute the pose of the person, by e.g.: http://www.ics.uci.edu/~dramanan/software/pose/. Perform classification relative to the skeleton of the pose
etc


IMPORTANT: Whenever you use someone's code / data, you need to cite the corresponding paper. You should not post this data online as it belongs to the paper mentioned above. You can post your results online, but please cite the paper.