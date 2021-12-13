function[ConfMat]=PNeural(Training_data)
%Data=readtable("Processed_Data.xlsx");
%Training_data=[Data.LB Data.AC Data.FM Data.UC Data.DL Data.DS Data.DP Data.ASTV Data.MSTV Data.ALTV Data.MLTV Data.Width Data.Min Data.Max Data.Nmax Data.Nzeros Data.Mode Data.Mean Data.Median Data.Variance Data.Tendency Data.NSP];
T= Training_data(:,22);
[f,scores]=fscmrmr(Training_data(:,1:21),T);
fs=f(1:10);
sel_feat=sort(fs);
New_feat=[];
for i=1:length(sel_feat)
    temp=Training_data(:,sel_feat(i));
    New_feat=[New_feat temp];
end

New_Training=[New_feat T];
Z=New_feat;
X1=[];
X2=[];
for i=1:size(New_feat,2)
    temp=New_feat(:,i);
    X1=[X1 temp(T==3)];
    X2=[X2 temp(T~=1)];
    
end
%Z=Training_data(:,1:21);
predictors = Z;
response  = T;
targ = zeros(size(T));
targ(T == 3) = 1;
x = Z';
t = targ';

%%
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)


% 
%%
% Plotting results 

pred_labels = zeros(size(y));
pred_labels(y >= 0.5) = 1;

%True_outs = 
pred_labels(pred_labels == 0)  = 2;
TestTarg = T;
TestTarg(T ~= 3) = 2;
TestTarg(T==3)=1;
ConfMat = my_confusion(pred_labels',TestTarg,2);
disp(ConfMat);
acc = sum(diag(ConfMat))/sum(sum(ConfMat));
disp("acc of Neural Network for P class after feature reduction: ")
disp(acc);
end