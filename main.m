clc;
clear;
close all;
Data=readtable("Processed_Data.xlsx");
Training_data=[Data.LB Data.AC Data.FM Data.UC Data.DL Data.DS Data.DP Data.ASTV Data.MSTV Data.ALTV Data.MLTV Data.Width Data.Min Data.Max Data.Nmax Data.Nzeros Data.Mode Data.Mean Data.Median Data.Variance Data.Tendency Data.NSP];      
figure(1);
ConfMat0=No_feature_reduction(Training_data);
confusionchart(ConfMat0);
title('SVM Classifier without applying feature reduction')
figure(2);
[ConfMat1,scores,f]=FMrMrSVM(Training_data);
confusionchart(ConfMat1);
title('SVM Classifier applying feature reduction')
figure(3);
Ranks=1:21;
stem(Ranks,f,'ro')
xlabel("Ranks");
ylabel("Features")
figure(4);
bar(scores(f))
xlabel('Predictor Ranks')
ylabel('Predictor Importance')
figure(5);
ConfMat2=NNeural(Training_data);
subplot(2,2,1)
confusionchart(ConfMat2);
title('Neural networks N as +class else all -class')
ConfMat3=SNeural(Training_data);
subplot(2,2,2)
confusionchart(ConfMat3);
title('             Neural networks S as +class else all -class')
ConfMat4=PNeural(Training_data);
subplot(2,2,3)
confusionchart(ConfMat4);
title('Neural networks P as +class else all -class')
