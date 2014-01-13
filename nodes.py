"""
    nodes.py - Construct a set of interpolation nodes with 
               low Lebesgue number on the triangle with 
               vertices {(-1,-1),(1,-1),(-1,1)}

    Copyright (C) 2014 Greg von Winckel

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated 
    documentation files (the "Software"), to deal in the 
    software without restriction, including without limitation 
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

    -----------------------------------------------------------

    Adapted from the paper

    Tim Warburton, "An Explicit Construction of Interpolation 
    Nodes on the Simplex," Journal of Engineering Mathematics, 
    November 2006, Volume 56, Issue 3, pp 247-262
    
    and the MATLAB code therein

    Also presented in the book
    Nodal Discontinuous Galerkin Methods
    Algorithms, Analysis, and Applications
    by Jan Hesthaven and Tim Warburton
    Springer Texts in Applied Mathematics, Vol 54, 2008


    Python implementation by Greg von Winckel on 
    Sun Sep 22 17:56:47 MDT 2013
    
    Modifications from original:
    1) Boundary points have been generalized to 
       Gegenbauer-Gauss-Lobatto nodes 
    2) Now supports up to p=21
    3) Nodes are now ordered as vertices, then edges, then 
       interior points using the counterclockwise indexing 
       convention

"""


import numpy as np
import orthopoly as op
from operator import mul

opt = [[0,0],[0,0],[34.5982022,-0.23422571],[-2.01869326e-05,5.29969190e-01],
       [1.12630113,0.36465773],[9.70630725e-01,1.34868622e-04],[1.70233407,0.61215954],
       [1.75767903,0.60108429],[1.73990095,0.54524112],[1.76606095,0.48266312],
       [1.718233,0.4185023],[1.75029272,0.36443944],[1.72542674,0.37479592],
       [1.68776479,0.28637813],[1.75802916,0.38338828],[1.65567624,5.21832816e-04],
       [1.78616273,-4.44607504e-04],[2.09374663,0.51318634],[2.01256402,0.51204339],
       [1.83243028,0.50097021],[2.08255411,0.65384728]]

def vertex_indices(p):
    return [0,1,2]

def edge_indices(p):
    d = np.arange(p-1)
    return list(3+d),list(p+2+d),list(2*p+1+d)        

def interior_indices(p):
    return range(3*p,(p+1)*(p+2)/2)


def nodes(p):
 
    alpha = opt[p-1][0]
    mu = opt[p-1][1]

    # Total number of nodes
    N = (p+1)*(p+2)/2

    # Compute Gegenbauer-Gauss-Lobatto nodes
    a,b = op.rec_jacobi(p+1,mu,mu)
    xl,_ = op.lobatto(a,b,-1,1)

    # Create uniformly distributed nodes on an equilateral triangle
    S = np.tile(range(p+1),(p+1,1))/float(p)
    L = np.zeros((N,3))
    L[:,0] = np.hstack([S[k:,k] for k in range(p+1)])
    L[:,1] = np.flipud(np.hstack([S[k,:k+1] for k in range(p+1)]))
    L[:,2] = np.hstack([S[k,:k+1] for k in range(p,-1,-1)])
    
    X = -L[:,1] + L[:,2] 
    Y = (2*L[:,0] - L[:,1] - L[:,2])/np.sqrt(3)


    # Compute blending function at each node for each edge
    blend1 = 4*L[:,1]*L[:,2]
    blend2 = 4*L[:,0]*L[:,2]
    blend3 = 4*L[:,0]*L[:,1]

    # Equidistant nodes
    xeq = np.linspace(-1,1,p+1)
    e = np.ones(p+1)

    # Distance between nodes
    dxeq = np.outer(xeq,e)-np.outer(e,xeq)+np.eye(p+1)
     
    def warpfactor(xnodes,xout):
        warp = np.zeros(N)
        
        for i in range(p+1):
            d = xnodes[i]-xeq[i]
            d *= reduce(mul,[(xout-xeq[j])/dxeq[i,j] for j in range(1,i)+range(i+1,p)],1)

            if i != 0:
                d /= -dxeq[i,0]
                
            if i != p:
                d /= dxeq[i,-1]
            warp += d
        return warp


    # Amount of warp for each node, for each edge         
    wf1 = warpfactor(xl,L[:,2]-L[:,1])
    wf2 = warpfactor(xl,L[:,0]-L[:,2])
    wf3 = warpfactor(xl,L[:,1]-L[:,0])

    # Combine warp and blend
    warp1 = blend1*wf1*(1+(alpha*L[:,0])**2)
    warp2 = blend2*wf2*(1+(alpha*L[:,1])**2)
    warp3 = blend3*wf3*(1+(alpha*L[:,2])**2)

    # Accumulate deformations associated with each edge
    theta = 2*np.pi/3
    X += warp1 + np.cos(theta)*warp2 + np.cos(2*theta)*warp3
    Y +=         np.sin(theta)*warp2 + np.sin(2*theta)*warp3

    # Map to lower right isoceles triangle
    M = np.array(((1.0/3,                 1.0/3,   1.0/3),
                  (-1.0/2,                    0,   1.0/2),
                  (-1/np.sqrt(12), 1/np.sqrt(3),  -1/np.sqrt(12))))

    vert = np.array(((-1,-1),(-1,1),(1,-1)))
    c = np.dot(M,vert)

    Xr = c[0,0] + c[1,0]*X + c[2,0]*Y
    Yr = c[0,1] + c[1,1]*X + c[2,1]*Y

    # Order nodes counterclockwise with vertices first, then edge nodes, 
    # then interior points
    v1 = 0
    v2 = p
    v3 = N-1
    edge1 = range(1,p)
    edge2 = [p*(k+1)-(k-1)*k/2 for k in range(1,p)]
    edge3 = [N-(k+1)*(k+2)/2 for k in range(1,p)]
    bpts = [v1,v2,v3] + edge1 + edge2 + edge3
    ipts = list(set(range(N)).difference(set(bpts)))
    order = bpts + ipts
    Xr = Xr[order]
    Yr = Yr[order]

    return Xr,Yr


if __name__ == '__main__':
#    from scipy.optimize import minimize
    import sys
    import matplotlib.pyplot as plt
    
     
    p = int(sys.argv[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x,y = nodes(p)
    labels = [str(k) for k in range(len(x))]
    ax.plot(x,y,'ro')
    for k in range(len(x)):
        ax.text(x[k],y[k],labels[k])
    plt.show()
