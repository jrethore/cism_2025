#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:11:50 2021

@author: julien_rethore
"""
import numpy as np
import scipy
import scipy.io
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter

def readDICmesh(inp):
    data = scipy.io.loadmat(inp, appendmat=False)
    xo=np.array(data['xo'])
    yo=np.array(data['yo'])
    X=np.c_[xo,yo]
    conn=np.array(data['conn'])-1
    return (X,conn)
    
def readDICDisp(inp,step=-1):
    data = scipy.io.loadmat(inp, appendmat=False)
    U=np.squeeze(np.array(data['U'])[:,step])
    return U

class FEModel:
    def __init__(self,d=2):

        self.dim = d # dimension of the mesh as an integer, 2 by default
        self.X = [] 
        self.conn = []
        self.material=[]
        
    def Assemble(self): 
        Nelems=self.conn.shape[0]
        Nnodes=self.X.shape[0]
        N=self.X[self.conn[:,1]]-self.X[self.conn[:,0]]
        l=(np.power(np.power(N[:,0],2)+np.power(N[:,1],2),0.5))
        self.W=scipy.sparse.spdiags(l,0,Nelems,Nelems)
        self.B=scipy.sparse.csr_matrix((N[:,0]/np.power(l,2),(np.arange(Nelems),np.squeeze(self.conn[:,1]))),shape=(Nelems,2*Nnodes))
        self.B+=scipy.sparse.csr_matrix((-N[:,0]/np.power(l,2),(np.arange(Nelems),np.squeeze(self.conn[:,0]))),shape=(Nelems,2*Nnodes))
        self.B+=scipy.sparse.csr_matrix((N[:,1]/np.power(l,2),(np.arange(Nelems),Nnodes+np.squeeze(self.conn[:,1]))),shape=(Nelems,2*Nnodes))
        self.B+=scipy.sparse.csr_matrix((-N[:,1]/np.power(l,2),(np.arange(Nelems),Nnodes+np.squeeze(self.conn[:,0]))),shape=(Nelems,2*Nnodes))

    def Solve(self,Ns,const_ddls,Uimp,Fext,U,verbosity = False):
        if verbosity: print('\nSolving...')
        Nelems=self.conn.shape[0]
        Nddl=U.shape[0]
        free=np.ones(Nddl)
        free[const_ddls]=0
        free_ddls=np.arange(Nddl)
        free_ddls=free_ddls[free>0]
        dS=self.material.GetInitTangent()
        K=(self.B.T)*(self.W*self.B*dS)        
        dU=np.zeros(Nddl)

        lfs=np.arange(1/Ns,1+1/Ns,1/Ns)
        itmax=100
        crit=1.e-6
        for lf in lfs:
            U[const_ddls]=Uimp[const_ddls]*lf
            iter=0
            ndU=1
            nres=1
            while (ndU > crit)&(iter<itmax):
                E=self.B.dot(U)
                S=self.material.GetStress(E)
                Fint=(self.B.T).dot(self.W.dot(S))
                Res=Fext*lf-Fint
                dU[free_ddls]=splinalg.spsolve(K[np.ix_(free_ddls,free_ddls)],Res[free_ddls])
                U+=dU
                iter+=1
                ndU=np.linalg.norm(dU)/np.linalg.norm(U)
                #nres=np.linalg.norm(Res[free_ddls])/np.linalg.norm(Fint)
                #print(nres)
            if verbosity: print('For load factor %5.3f after %d iterations R/Fext = %8.3e dU/U=%8.3e'% (lf,iter,nres,ndU))
        return Fint
    def show_field(self,fig, axsi,  
            A, name = ""):
            segments = []
            if len(A)==self.X.shape[0]:
                values = []
            else:
                values=A
            mini_ = np.mean(A)-2*np.std(A)
            maxi_ = np.mean(A)+2*np.std(A)
            for e in self.conn:
                n1, n2 = e
                p1, p2 = self.X[n1], self.X[n2]
                segments.append([p1, p2])
                # couleur moyenne des extrémités (champ élémentaire)
                if len(A)==self.X.shape[0]:
                    values.append((A[n1] + A[n2]) / 2)


            segments = np.array(segments)
            values = np.array(values)

            # Création de la plasma
            lc = LineCollection(segments, array=values, cmap='plasma', linewidths=2)

            # Tracé
            h = axsi.add_collection(lc)
            axsi.set_title(name)
            axsi.set_xlabel('X')
            axsi.set_ylabel('Y')
            axsi.axis('equal')
            axsi.set_xlim(( np.min(self.X[:,0]),np.max(self.X[:,0]) ))
            axsi.set_ylim(( np.min(self.X[:,1]),np.max(self.X[:,1])*1.2 ))
            h.set_clim(vmin=mini_,vmax=maxi_)
            cbar = fig.colorbar(h, ax=axsi,orientation='horizontal')
            h.set_clim(np.quantile(A,0.05), np.quantile(A,0.95))
            #cbar.set_ticks([np.quantile(A,0.05),np.quantile(A,0.95)])
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.0e"))
            return h,cbar

    def show_mesh(self,fig, axsi):
            h = plt.plot(self.X[self.conn,0].T,self.X[self.conn,1].T,'b-')
            axsi.set_title('Mesh')
            axsi.set_xlabel('X')
            axsi.set_ylabel('Y')
            axsi.axis('equal')
            axsi.set_xlim(( np.min(self.X[:,0]),np.max(self.X[:,0]) ))
            axsi.set_ylim(( np.min(self.X[:,1]),np.max(self.X[:,1])*1.2 ))



class MatModel:
    def __init__(self,s=1):

        self.K = []
        self.Eo = []
        self.scale = s
    def GetStress(self,E):
        S=self.K*(1-np.exp(-np.abs(E)/np.abs(self.Eo)))*np.sign(E)*self.scale
        return S
    def GetInitTangent(self,):
        S=self.K/self.Eo*self.scale
        return S
    def GetTangent(self,E):
        S=self.K*np.exp(-np.abs(E)/self.Eo)/self.Eo*self.scale
        return S

