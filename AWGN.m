function Noise = AWGN(x,No)
%
% Inputs:
%   x: signal
%   No:     2 times the noise variance
% Outputs:
%   Noise:      
%%% WRITE YOUR CODE HERE
Noise = normrnd(0,No,size(x)) ; 
%%%