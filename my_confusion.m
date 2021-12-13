function Confmat = my_confusion(Pred_labels,T,M)
Confmat=zeros(M);
for i=1:length(T)
  Confmat(T(i),Pred_labels(i))=Confmat(T(i),Pred_labels(i))+1;
end
end