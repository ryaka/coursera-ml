function [ optimC, optimSigma ] = findOptimal( X, y, Xval, yval )
%FINDOPTIMAL Summary of this function goes here
%   Detailed explanation goes here

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
Sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

accuracy = -1;

[p,q] = meshgrid(C, Sigma);
combinations = [p(:) q(:)];

for index = 1:size(combinations,1)
    c = combinations(index,1);
    sigma = combinations(index,2);
    
    model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma));
    pred = svmPredict(model, Xval);
    
    modelAccuracy = mean(double(pred ~= yval));
    if accuracy < 0 || modelAccuracy < accuracy
        accuracy = modelAccuracy;
        optimC = c;
        optimSigma = sigma;
    end
end

end

