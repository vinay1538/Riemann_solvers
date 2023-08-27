#AUSM
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#function to calculate rho,u,p,e,H
def rho_u_p_e_h(U,i):
  gamma=1.4
  rho,u,p=U[0][i],U[1][i]/U[0][i],(U[2][i]-U[1][i]**2/(U[0][i]*2))*(gamma-1)
  e=0.5*(uL)**2+p/((gamma-1)*rho)
  H=e+(p/rho)
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

F_plus = np.array([[0.0, 0.0, 0.0]] * N)
F_minus = np.array([[0.0, 0.0, 0.0]] * N)
while t<0.5:
    u=np.zeros((3,N))
    delta=np.zeros(N)
    Lambda=np.zeros(N)
    Lambda_max=0
    f_plus=np.zeros((N,3))
    f_minus=np.zeros((N,3))
    #ghost cell technique
    U[0][0]=U[0][1]
    U[0][N-1]=U[0][N-2]
    U[2][0]=U[2][1]
    U[2][N-1]=U[2][N-2]
    U[1][0]=-U[1][1]
    U[1][N-1]=-U[1][N-2]
    for i in range(0,N):
       #rho,u,p,e,H values
        rhoL,uL,pL,eL,HL=rho_u_p_e_h(U,i)
        
        a=sqrt(gamma*pL/rhoL)
        
        #eigen values
        Lambda=np.array([uL-a,uL,uL+a])
        Lambda_max=max(Lambda_max,max(max(Lambda),-min(Lambda)))
    
        # + and - fluxes at cell
        m=uL/a
        if m<=-1:
            m_plus,m_minus=0,m
            p_plus,p_minus=0,pL
        elif -1<m<1:
            m_plus,m_minus=((m+1)/2)**2,-((m-1)/2)**2
            p_plus,p_minus=pL*0.5*(1+m),pL*0.5*(1-m)
        elif m>=1:
            m_plus,m_minus=m,0
            p_plus,p_minus=pL,0
        #HL=HL+0.5*uL**2
        fc=np.array([ rhoL*a, rhoL*uL*a, rhoL*HL*a ])
        fp_plus=np.array([0,p_plus,0])
        fp_minus=np.array([0,p_minus,0])
        
        for j in range(3):
          F_plus[i][j]=m_plus*fc[j]+fp_plus[j]
          F_minus[i][j]=m_minus*fc[j]+fp_minus[j]
    del_t = cfl*del_x / Lambda_max
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

axs[0, 0].plot(x_plot, rho,label="AUSM)")
axs[0, 0].set_title("density")
axs[0, 0].set(xlabel='position', ylabel='Density')
axs[0, 0].legend(loc='best')


axs[1, 0].plot(x_plot,u,label="AUSM")
axs[1, 0].set_title("velocity")
axs[1, 0].set(xlabel='position', ylabel='velocity')
axs[1, 0].legend(loc='upper left')


axs[0, 1].plot(x_plot,p,label="AUSM")
axs[0, 1].set_title("pressure")
axs[0, 1].set(xlabel='position', ylabel='Pressure')
axs[0, 1].legend(loc='best')


axs[1, 1].plot(x_plot, e,label="AUSM")
axs[1, 1].set_title("Internal energy")
axs[1, 1].set(xlabel='position', ylabel='Internal energy')
axs[1, 1].legend(loc='best')
fig.tight_layout()
plt.show()
