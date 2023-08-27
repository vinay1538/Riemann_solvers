#roe solver without entropy fix
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reding the file for Exact values
file=np.loadtxt("StandardSod_0_15.txt",skiprows=1)
x_std=file[:,0]
rho_std=file[:,1]
u_std=file[:,2]
p_std=file[:,3]
e_std=file[:,4]

#function to calculate rho,u,p,e,H
def rho_u_p_e_h(U,i):
  gamma=1.4
  rho,u,p=U[0][i],U[1][i]/U[0][i],(U[2][i]-U[1][i]**2/(U[0][i]*2))*(gamma-1)
  e=0.5*(uL)**2+p/((gamma-1)*rho)
  H=e+p/rho
  return rho,u,p,e,H

gamma = 1.4
N = 400
cfl = 0.8
start_point = 0
end_point =1

#given Conditions
rhoL, pL, uL = 1.0, 1.0, 0.0
rhoR, pR, uR = 0.125, 0.1, 0.0

eL=0.5*(uL)**2+pL/((gamma-1)*rhoL)
eR=0.5*(uR)**2+pR/((gamma-1)*rhoR)

F=np.zeros(3) #initializing flux at centre of cell
f=np.zeros((3,N))#initializing flux at faces
U=np.zeros((3,N))# initializing U matrix

x_plot=np.linspace(start_point,end_point,N)

for i in range(N):
    if x_plot[i]<0.5:
      U[0][i]=rhoL
      U[1][i]=rhoL*uL
      U[2][i]=rhoL*eL
    else:
      U[0][i]=rhoR
      U[1][i]=rhoR*uR
      U[2][i]=rhoR*eR

delta = np.zeros(N)
Lambda = np.zeros(N)
del_x = end_point / N
t = 0
while t<=0.5:

    Lambda_max=0

    for i in range(N-1):
      #rho,u,p,e,H values
        rhoL,uL,pL,eL,HL = rho_u_p_e_h(U,i)
        rhoR,uR,pR,eR,HR = rho_u_p_e_h(U,i+1)

        # telda values
        u_telda = (sqrt(rhoL)*uL + sqrt(rhoR)*uR)/(sqrt(rhoR)+sqrt(rhoL))
        H_telda=(sqrt(rhoL)*HL + sqrt(rhoR)*HR)/(sqrt(rhoR)+sqrt(rhoL))
        a_telda=(gamma-1)*(H_telda-0.5*u_telda**2)

        #eigen values
        Lambda = np.array([(u_telda-a_telda) , (u_telda) , (u_telda+a_telda)])
        Lambda_max=max(Lambda_max,max(max(Lambda),-min(Lambda)))   

        #Delta_U values
        delta_u1=U[0][i+1]-U[0][i]
        delta_u2=U[0][i+1]*U[1][i+1]-U[0][i]*U[1][i]
        delta_u3=U[0][i+1]*eR-U[0][i]*eL

        #delta_i values
        delta2=(gamma-1)*(delta_u1*(H_telda-u_telda**2)+u_telda*delta_u2-delta_u3)/a_telda**2
        delta1=0.5*( delta_u1*(u_telda+a_telda)-delta_u2-a_telda*delta2 )/a_telda
        delta3= delta_u1-(delta1+delta2)
        delta=np.array([delta1,delta2,delta3])

        #eigen vectors
        k1=[1,  u_telda-a_telda, H_telda - u_telda*a_telda]
        k2=[1,  u_telda, 0.5*u_telda**2]
        k3=[1,  u_telda+a_telda, H_telda + u_telda*a_telda]

        K = np.array([k1,k2,k3])
        K = np.transpose(K)

        F[0] = rhoL*uL
        F[1] = (rhoL*(uL**2)+pL)
        F[2] = uL*(U[2][i]+pL)
        #calculating fluxes at faces
        for j in range(3):
            sum = 0
            for k in range(3):
                if(Lambda[k]>0):
                    break
                sum += Lambda[k]*delta[k]*K[j][k]
            f[j][i] = F[j]+sum

    del_t=cfl*del_x/Lambda_max

    for i in range(1,N-1):
        for j in range(3):
            U[j][i] =  U[j][i] - (del_t/del_x) * (f[j][i] - f[j][i-1])

    t+=del_t

#Getting final answers
rho = np.copy (U[0])
u = np.copy (U[1]/U[0])
p = np.copy ((U[2]-U[1]**2/(U[0]*2))*(gamma-1))
e = np.copy (p/((gamma-1)*rho))
#plotting
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x_std,rho_std,label='Analytic')
axs[0, 0].scatter(x_plot, rho,marker='.',label="Roe approximate")
axs[0, 0].set_title("density")
axs[0, 0].set(xlabel='position', ylabel='Density')
axs[0,0].legend(loc='best')

axs[1, 0].plot(x_std,u_std,label='Analytic')
axs[1, 0].scatter(x_plot,u,marker='.',label="Roe approximate")
axs[1, 0].set_title("velocity")
axs[1, 0].set(xlabel='position', ylabel='velocity')
axs[1, 0].legend(loc='upper left')

axs[0, 1].plot(x_std,p_std,label='Analytic')
axs[0, 1].scatter(x_plot,p,marker='.',label="Roe approximate")
axs[0, 1].set_title("pressure")
axs[0, 1].set(xlabel='position', ylabel='Pressure')
axs[0, 1].legend(loc='best')

axs[1, 1].plot(x_std,e_std,label='Analytic')
axs[1, 1].scatter(x_plot, e,marker='.',label="Roe approximate")
axs[1, 1].set_title("Internal energy")
axs[1, 1].set(xlabel='position', ylabel='Internal energy')
axs[1, 1].legend(loc='best')
fig.tight_layout()
plt.show()
