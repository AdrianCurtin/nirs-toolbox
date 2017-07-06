function [R,p,dfe]=ar_xcorr(data,modelorder,maxlags,robust_flag)

if(nargin<4)
    robust_flag=true;
end

if(nargin<3)
    maxlags=[];
end

if(nargin<2 || isempty(modelorder))
    modelorder=20;
end

if(~isempty(strfind(class(data),'.core.Data')))
    Fs=data.Fs;
    data=data.data;
else
    Fs=1;
end

if(isstr(modelorder))
    p = Fs*str2num(modelorder(1:strfind(modelorder,'x')-1));
else
    p=modelorder;
end

if(isstr(maxlags))
    maxlags = ceil(Fs*str2num(maxlags(1:strfind(maxlags,'x')-1)));
end

if(~isreal(data))
    mask=(imag(data)>0);
    
else
    mask=ones(size(data));
end

[yfilt,f] = nirs.math.innovations(real(data),p);

if(robust_flag)
    [R,p]=nirs.math.robust_xcorrcoef(yfilt,maxlags,mask);
else
    [R,p]=nirs.math.xcorrcoef(yfilt,maxlags,mask);
end

dfe = mean(sum(mask)) - 2;
