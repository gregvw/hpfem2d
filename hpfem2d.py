# -*- coding: utf-8  -*-

"""
    hpyfem2d.py - Program for generating 2D hp finite 
                  element trial functions and their
                  derivatives

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

import numpy as np
import orthopoly as op
from nodes import nodes, vertex_indices, edge_indices, interior_indices
from pkdo import pkdo
from triquad import triquad
from numpy.linalg import inv

class hpfem2d(object):

    def __init__(self,p):
        """ 
        Form a basis generating object based on the (p+1)(p+2)/2 
        interpolation nodes
        """
        self.p = p

        # Construct the interpolation nodes
        self.x, self.y = nodes(self.p)

        # Form the PKDO Vandermonde on the nodes 
        V,_,_ = pkdo(self.p,self.x,self.y)

        # Compute the inverse of the interpolation Vandermonde
        self.Vi = inv(V)

    def getInteriorTrial(self,q):
        """
        Evaluate the nodal interpolating functions and their x and y
        derivatives on a quadrature grid of q^2 points
        """
        
        # Generate interior quadrature grid
        xq,yq,wq = triquad(q)
        
        # Compute Vandermondes PKDO polynomials and their derivatives on
        # quadrature grid
        V,Vx,Vy = pkdo(self.p,xq,yq)

        # Trial functions
        L = np.dot(V,self.Vi)

        # x derivative of trial functions
        Lx = np.dot(Vx,self.Vi)

        # y derivative of trial functions
        Ly = np.dot(Vy,self.Vi)

        return xq,yq,wq,L,Lx,Ly

    def getBoundaryTrial(self,q,edge):
        """
        Evaluate the nodal interpolating functions along one of the edges
        using q Legendre Gauss nodes

        """

        # Gauss quadrature recursion coefficients
        a,b = op.rec_jacobi(q,0,0)

        # Legendre Gauss nodes and weights
        t,wt = op.gauss(a,b)

        # Affine map of [-1,1] to the appropriate triangle edge
        xdict = {0:t,1:-t,2:-np.ones(q)}
        ydict = {0:-np.ones(q),1:t,2:-t}

        # Evaluate PKDO Vandermonde on the quadrature grid
        V,_,_ = pkdo(self.p,xdict[edge],ydict[edge])

        # Evaluate 2D Lagrange interpolants edge
        L = np.dot(V,self.Vi)

        return xdict[edge],ydict[edge],wt,L 



def manufactured_solution(expression):
    """ Evaluate a string for the exact symbolic solution and
        create numerical function handles for all of the terms
        needed to reconstruct it by solving the BVP """

    from sympy import *
    
    # Define symbolic variables for manufactured solution
    x,y = symbols('x,y')

    # Exact symbolic solution
    u = eval(expression)

    # Partial derivatives
    ux = diff(u,x)
    uy = diff(u,y)

    # symbolic forcing function 
    f = -diff(ux,x)-diff(uy,y)

    # Return list of numerical function handles
    return [lambdify([x,y],fun,"numpy") for fun in [u,ux,uy,f]]




if __name__ == '__main__':
    """ 
    Solve the Poisson equation with unit forcing on the 
    lower right triangle with Dirichlet (0), Neumann (1), and Robin (2)
    conditions
    """

    from scipy.linalg import solve
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Max polynomial order  
    p = 20 
 
    # Indices of interior and boundary points
    idex = interior_indices(p)
    edex = edge_indices(p)

    # Instantiate FEM basis generator for this order
    FEM = hpfem2d(p)        

    # Get function handles for the manufactured solution 
    u,ux,uy,f = manufactured_solution("cos(pi*(x-y)) + sin(pi*(x+y))")

    # Get interior points and basis functions
    xq,yq,wq,L,Lx,Ly = FEM.getInteriorTrial(p)
    
    # Get boundary quadrature and basis functions
    x1,y1,w1,L1 = FEM.getBoundaryTrial(p,1)
    x2,y2,w2,L2 = FEM.getBoundaryTrial(p,2)

    # Inner product over the elemental interior
    def iprod(A,B):
        return np.dot(wq*A.T,B)

    # Interpolation points
    x,y = FEM.x,FEM.y 

    # Total number of nodes
    N = len(x)

    # Evaluate the exact solution on edge 0 - including the vertex nodes
    # because this side has a Dirichlet condition
    e0 = [0,1] + edex[0] 
 
    a = u(x[e0],y[e0])

    # Evaluate the normal derivative on edge 1
    b = ux(x1,y1) + uy(x1,y1)

    # Evaluate the solution plus normal derivative on edge 2
    c = u(x2,y2) - ux(x2,y2)

    # Compute load vector
    fhat = iprod(L,f(xq,yq))

    # Integrate inhomogeneous boundary terms against test functions
    bhat = np.dot(w1*L1.T,b)
    chat = np.dot(w2*L2.T,c)

    # Surface matrix for Robin condition on edge 2
    S = np.dot(w2*L2.T,L2)

    # Stiffness matrix
    K = iprod(Lx,Lx)+iprod(Ly,Ly)

    # Left-hand-side
    LHS = K + S

    # Computed solution
    psi = np.zeros(N)
    
    # Set Dirichlet data
    psi[e0] = a

    # Right-hand-side
    rhs = fhat + bhat + chat - np.dot(LHS,psi)

    # Solve for interior points, and points on edges 1 and 2, and vertex 2
    dex = idex+edex[1]+edex[2]+[2]
    psi[dex] = solve(LHS[dex,:][:,dex],rhs[dex])

    fig = plt.figure()
    ax1= fig.add_subplot(121,projection='3d')
    ax2= fig.add_subplot(122,projection='3d')

    ax1.plot_trisurf(x,y,psi,cmap=plt.cm.CMRmap)
    ax1.set_title('Computed Solution')

    ax2.plot_trisurf(x,y,u(x,y),cmap=plt.cm.CMRmap)
    ax2.set_title('Exact Solution')
    plt.show()


    

