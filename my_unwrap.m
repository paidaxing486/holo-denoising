function ignew=my_unwrap(z)
[M,N]=size(z);
temp1(1:M-1,1:N)=z(2:M,1:N); 
temp1(M,1:N)=z(M,1:N);

temp2(1:M,1:N-1)=z(1:M,2:N);
temp2(1:M,N)=z(1:M,N);

phasediffx=temp1-z;
phasediffy=temp2-z;

for i=1:M
   for j=1:N
      if phasediffx(i,j)>pi 
         phasediffx(i,j)=phasediffx(i,j)-2*pi;
      elseif phasediffx(i,j)<-pi
            phasediffx(i,j)=phasediffx(i,j)+2*pi;
      end
      if phasediffy(i,j)>pi 
         phasediffy(i,j)=phasediffy(i,j)-2*pi;
      elseif phasediffy(i,j)<-pi
            phasediffy(i,j)=phasediffy(i,j)+2*pi;
      end
   end
end
temp3(1,1:N)=0;
temp3(2:M,1:N)=phasediffx(1:M-1,1:N);
temp4(1:M,1)=0;
temp4(1:M,2:N)=phasediffy(1:M,1:N-1);
Rho=phasediffx-temp3+phasediffy-temp4;
%%%%%%%%%%%dct2-idct2
Rhonew=dct2(Rho);  
   for i=1:M
      for j=1:N
         if (i==1)&(j==1)
            ig(i,j)=Rhonew(i,j);
         else
            ig(i,j)=Rhonew(i,j)/(2*(cos(pi*(i-1)/M)+cos(pi*(j-1)/N)-2));
         end
      end
   end
   ignew=idct2(ig);
%  ignew=revised(z,ignew);
 return  



