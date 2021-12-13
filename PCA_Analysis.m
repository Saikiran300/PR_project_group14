Data=readtable("Processed_Data.xlsx");

Training_data=[Data.LB Data.AC Data.FM Data.UC Data.DL Data.DS Data.DP Data.ASTV Data.MSTV Data.ALTV Data.MLTV Data.Width Data.Min Data.Max Data.Nmax Data.Nzeros Data.Mode Data.Mean Data.Median Data.Variance Data.Tendency Data.NSP];
T= Training_data(:,22);
[coeff,score,latent,tsquared,explained,mu] = pca(Training_data(:,1:21));
explained
bar(explained)