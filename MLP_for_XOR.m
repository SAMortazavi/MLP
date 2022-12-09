clc
clear 
close all
w11=random_Weight(3);
w12=random_Weight(3);
w21=random_Weight(3);
x1=[0 1 0 1];
x2=[0 0 1 1];
Z=[];
y=[0 1 1 0];
eta=0.03;
E=1;
yp=[];
j=1;
MSE=[];
while E>0.01
    
    E=0;
    for i=1:3 
        z1=act_sigmoid(inner_product([x1(i) x2(i) 1],w11));
        z2=act_sigmoid(inner_product([x1(i) x2(i) 1],w12));
        Z=[z1 z2 1];
        yp(i)=act_sigmoid(inner_product(Z,w21));
        E=E+0.5*(y(i)-yp(i))^2;
        w11=update_Weight1(w11,Z,y(i),yp(i),[x1(i) x2(i) 1],eta);
        w12 =update_Weight1(w12,Z,y(i),yp(i),[x1(i) x2(i) 1],eta);
        w21=update_Weight2(w21,y(i),yp(i),Z,eta);
    end
    disp(E);
    MSE(j)=E;
    j=j+1;
end 
plot(MSE,'r.');
function W1=random_Weight(n)
W1=randn(1,n);
return
end
function output1=Out_first(x,w)
out=inner_product(w,x);
output1=act_sigmoid(out);
return
end
function out_sigmoid=act_sigmoid(x)
out_sigmoid=(1/(1+exp(-x)));
return
end
function def_sig=def_sigmoid(y)
 def_sig=2*exp(-y)/(1+exp(-y))^2;
   return
end
function out_dot=inner_product(w,x)
out_dot=0;
for i=1:length(w)
    out_dot=out_dot+x(i)*w(i);
end
return
end
function W11=update_Weight1(w11,z,y,yp,x,eta)
W11=w11+eta*2*yp*(y-yp)*(1-yp)*z.*(1-z).*x/3;
end
function W21=update_Weight2(w21,y,yp,z,eta)
W21=w21+eta*2*(y-yp)*yp*(1-yp)*z/3;
end