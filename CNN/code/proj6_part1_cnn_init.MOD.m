function net = proj6_part1_cnn_init()
%code for Computer Vision, Georgia Tech by James Hays
%based of the MNIST example from MatConvNet

rng('default');
rng(0);

% constant scalar for the random initial network weights. You shouldn't
% need to modify this.
f=1/100; 


% Let's make our network deeper by adding an additional convolutional layer 
% in proj6_part1_cnn_init.m. In fact, we probably don't want to add just a 
% convolutional layer, but another max-pool layer and relu layer, as well. 
% For example, you might insert a convolutional layer after the existing 
% relu layer with a 5x5 spatial support followed by a max-pool over a 3x3 
% window with a stride of 2. You can reduce the max-pool window in the 
% previous layer, adjust padding, and reduce the spatial resolution of the 
% final layer until vl_simplenn_display(net, 'inputSize', [64 64 1 50]), 
% which is called at the end of proj6_part1_cnn_init() shows that your 
% network's final layer (not counting the softmax) has a data size of 1 and 
% a data depth of 15. You also need to make sure that the data depth output 
% by any channel matches the data depth input to the following channel. For 
% instance, maybe your new convolutional layer takes in the 10 channels of 
% the first layer but outputs 15 channels. The final layer would then need 
% to have its weights initialized accordingly to account for the fact that 
% it operates on a 15 channel image instead of a 10 channel image. 

net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(9,9,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;

net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,10, 'single'), zeros(1,10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv2') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', 0.5) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(11,11,10,15, 'single'), zeros(1, 15, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'fc1') ;
                      
% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

net = vl_simplenn_tidy(net);

% %You can insert batch normalization layers here
% net = insertBnorm(net, 1)

% Visualize the network
vl_simplenn_display(net, 'inputSize', [64 64 1 50])


% --------------------------------------------------------------------
function net = insertBnorm(net, layer_index)
% --------------------------------------------------------------------
assert(isfield(net.layers{layer_index}, 'weights'));
ndim = size(net.layers{layer_index}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05]) ;
net.layers{layer_index}.weights{2} = [] ;  % eliminate bias in previous conv layer
net.layers = horzcat(net.layers(1:layer_index), layer, net.layers(layer_index+1:end)) ;



