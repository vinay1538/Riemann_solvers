#stegar 
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##file=np.loadtxt("StandardSod_0_15.txt",skiprows=1)
##x_std=file[:,0]
##rho_std=file[:,1]
##u_std=file[:,2]
##p_std=file[:,3]
##e_std=file[:,4]

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
del_x = end_point / N

rhoL, pL, uL = 1.0, 1.0, 0.0
rhoR, pR, uR = 0.125, 0.1, 0.0

eL=0.5*(uL)**2+pL/((gamma-1)*rhoL)
eR=0.5*(uR)**2+pR/((gamma-1)*rhoR)

xpoints=np.linspace(start_point,end_point,N+1)
F = np.array([0.0, 0.0, 0.0])
f = np.array([[0.0, 0.0, 0.0]] * N)

U=np.zeros((3,N))
for i in range(N):
    if xpoints[i]<0.5:
      U[0][i]=rhoL
      U[1][i]=rhoL*uL
      U[2][i]=rhoL*eL
    else:
      U[0][i]=rhoR
      U[1][i]=rhoR*uR
      U[2][i]=rhoR*eR
t=0
itr=0
F_plus = np.array([[0.0, 0.0, 0.0]] * N)
F_minus = np.array([[0.0, 0.0, 0.0]] * N)
while t<0.5:
    u,u_old=np.zeros((3,N)),np.zeros((3,N))
    delta=np.zeros(N)
    Lambda=np.zeros(N)
    Lambda_max=0
    f_plus=np.zeros((N,3))
    f_minus=np.zeros((N,3))
    for i in range(N):
       #rho,u,p,e,H values
        rhoL,uL,pL,eL,HL=rho_u_p_e_h(U,i)
  
        a=(gamma*pL/rhoL)**0.5
        #eigen values
        Lambda=np.array([uL-a,uL,uL+a])
        Lambda_max=max(Lambda_max,max(max(Lambda),-min(Lambda)))
        #eigen values for + and - 
        lambda1_plus,lambda2_plus,lambda3_plus=max(Lambda[0],0),max(Lambda[1],0),max(Lambda[2],0)
        lambda1_minus,lambda2_minus,lambda3_minus=min(Lambda[0],0),min(Lambda[1],0),min(Lambda[2],0)
        # + and - fluxes at cell
        f_plus[i][0]=rhoL/(2*gamma) *(lambda1_plus+2*(gamma-1)*lambda2_plus+lambda3_plus)
        f_plus[i][1]=rhoL/(2*gamma) *(lambda1_plus*Lambda[0]+2*(gamma-1)*Lambda[1]*lambda2_plus+lambda3_plus*(Lambda[2]))
        f_plus[i][2]=rhoL/(2*gamma) *(lambda1_plus*(HL-uL*a)+(gamma-1)*uL**2*lambda2_plus+lambda3_plus*(HL+uL*a))
        f_minus[i][0]=rhoL/(2*gamma) *(lambda1_minus+2*(gamma-1)*lambda2_minus+lambda3_minus)
        f_minus[i][1]=rhoL/(2*gamma) *(lambda1_minus*Lambda[0]+2*(gamma-1)*Lambda[1]*lambda2_minus+lambda3_minus*(Lambda[2]))
        f_minus[i][2]=rhoL/(2*gamma) *(lambda1_minus*(HL-uL*a)+(gamma-1)*uL**2*lambda2_minus+lambda3_minus*(HL+uL*a))
        
        for j in range(3):
          F_plus[i][j]=f_plus[i][j]
          F_minus[i][j]=f_minus[i][j]
    del_t=cfl*del_x/Lambda_max
    for i in range(1,N-1):
        for j in range(3):
            U[j][i]=U[j][i]-(del_t/del_x)*((F_plus[i][j]+F_minus[i+1][j])-(F_plus[i-1][j]+F_minus[i][j]))
    t+=del_t
x_plot=np.linspace(start_point,end_point,N)
#Getting final answers
rho=np.copy(U[0])
u=np.copy(U[1]/U[0])
p=np.copy((U[2]-U[1]**2/(U[0]*2))*(gamma-1))
e=np.copy(p/((gamma-1)*rho))

#plotting
fig, axs = plt.subplots(2, 2)
#axs[0, 0].plot(x_std,rho_std,label='Analytic')
axs[0, 0].scatter(x_plot, rho,marker='.',label="Steger warming")
axs[0, 0].set_title("density")
axs[0, 0].set(xlabel='position', ylabel='Density')
axs[0,0].legend(loc='best')

#axs[1, 0].plot(x_std,u_std,label='Analytic')
axs[1, 0].scatter(x_plot,u,marker='.',label="Steger warming")
axs[1, 0].set_title("velocity")
axs[1, 0].set(xlabel='position', ylabel='velocity')
axs[1, 0].legend(loc='best')

#axs[0, 1].plot(x_std,p_std,label='Analytic')
axs[0, 1].scatter(x_plot,p,marker='.',label="Steger warming")
axs[0, 1].set_title("pressure")
axs[0, 1].set(xlabel='position', ylabel='Pressure')
axs[0, 1].legend(loc='best')

#axs[1, 1].plot(x_std,e_std,label='Analytic')
axs[1, 1].scatter(x_plot, e,marker='.',label="Steger warming")
axs[1, 1].set_title("Internal energy")
axs[1, 1].set(xlabel='position', ylabel='Internal energy')
axs[1, 1].legend(loc='best')
fig.tight_layout()
plt.show()

