function rect_bits= decision(x)
rect_bits=zeros(1,length(x));
for i=1:1:length(x)
     if x(i)>0
         rect_bits(i)=1;
     else
         rect_bits(i)=0;
     end
end