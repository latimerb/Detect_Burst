function [ tpr,fpr ] = ROC( target,output )
if ~isvector(target)
    error('Target must be a vector.');
end
if ndims(output)>2
    error('Output must be a matrix.');
end
N = length(target);
dim = find(size(target)~=1,1);
if size(output,dim)~=N
    error('Target and output dimension must match.');
end
if dim==2
    target = target';
    output = output';
end

P = sum(target);
tpr = sum(bsxfun(@and,target,output),1)/P;
fpr = sum(bsxfun(@and,~target,output),1)/(N-P);

hold on;
plot([0,1],[0,1],'r:');
plot(fpr,tpr,'b');
xlabel('False Positive Rate');
ylabel('True Positive Rate');

end

