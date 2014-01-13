# -*- coding: utf-8  -*-
"""
pkdo.py - Generate the Proriol-Koornwinder-Dubiner-Owens
          polynomials and their derivatives

Copyright (C) 2014 Greg von Winckel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the 
Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to 
do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY 
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS 
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import orthopoly as op
import numpy as np

def pkdo(p,x,y):
    """
    Evaluate the Vandermonde matrix of PKDO polynomials up to order on the set of points (x,y) 
    as well as the Vandermondes containing the x and y derivatives of the polynomials
    """
    Nx = len(x)
    Np = (p+1)*(p+2)/2
    s = -np.ones(Nx)
    t = y
    dex = np.abs(y-1) > 1e-10
    s[dex] = 2*(1+x[dex])/(1-y[dex])-1

    # Vandermonde
    V = np.zeros((Nx,Np))

    # Derivative w.r.t. x
    Vx = np.zeros((Nx,Np))

    # Derivative w.r.t. y
    Vy = np.zeros((Nx,Np))

    ll = 0

    tfact0 = np.zeros(Nx)
    tfact  = np.ones(Nx)

    Ps = op.jacobi(p,0,0,s)
    Psder = op.jacobiD(p,0,0,s)
 
    for jj in range(p+1):

        Pt = op.jacobi(p+1,2*jj+1,0,t)
        Ptder = op.jacobiD(p+1,2*jj+1,0,t)

        for kk in range(p+1-jj):
            
            nfact = np.sqrt((jj+0.5)*(jj+kk+1))                
            V[:,ll] = nfact*Ps[:,jj]*Pt[:,kk]*tfact
            Vx[:,ll] = Psder[:,jj]*Pt[:,kk]
            Vy[:,ll] = Vx[:,ll]*(1+s)/2

            u = Ptder[:,kk]*tfact

            if jj>0:

                Vx[:,ll] *= nfact*tfact0
                Vy[:,ll] *= tfact0
                u -= 0.5*jj*Pt[:,kk]*tfact0

            Vy[:,ll] += Ps[:,jj]*u
            Vy[:,ll] *= nfact
              
            ll += 1
        
        tfact0 = tfact        
        tfact = tfact0*(1-t)/2

    return V,Vx,Vy
