from nodes import nodes
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sys


def edgeMap(xu,u1,u2,v1,v2):

    """
    Map the set of points xu, which lie on the line segment [u1,u2], 
    onto the points xv, which lie on the line segment [v1,v2]
    """  

    n = xu.shape[0]
    e = np.ones(n)  
    t = (np.dot(xu,u1-u2)-e*np.dot(u2,u1-u2))/np.dot(u1-u2,u1-u2)
    xv = np.outer(e,v2)+np.outer(t,v1-v2)

    return xv

def interiorMap(xu,v):
    """
    Map the set of points xu which lie on the 'unit' reference triangle with 
    vertices {(-1,-1),(1,-1),(-1,1)} 
    """
    n = xu.shape[0]
    e = np.ones(n)
    M = np.array(((1,-1,-1),(1,1,-1),(1,-1,1))) 
    ab = np.linalg.solve(M,v)
    exu = np.vstack((e,xu.T)).T
    xv = np.dot(exu,ab)
    return xv




class assembler(object):
    """
    Finite element matrix assembly object using the 
    """


    def __init__(self,triang,p):
        
        self.p     = p                 # Maximal polynomial order
        self.Np    = (p+1)*(p+2)/2     # Number of points per element
        self.Npi   = (p-1)*(p-2)/2     # Number of interior points per element
        self.t     = triang.triangles  # List of elements by vertices
        self.nbr   = triang.neighbors  # Adjacent elements
        self.edges = triang.edges      # List of edges which connect vertices
        self.Ne    = len(self.edges)   # Number of edges
        self.xv    = triang.x          # Vertex nodes (x,y)
        self.yv    = triang.y           
        self.Nt    = len(self.t)       # Number of triangles
        self.Nv    = len(self.xv)      # Number of vertices 

        # Interpolation points on the reference triangle
        xn,yn = nodes(p)

        xy = np.array(zip(xn,yn))       

        plt.plot(self.xv,self.yv,'ro')   

        # for edge dofs: construct local-to-global index for edges
        # symmetrize pair of vertices per edge
        L = [tuple(self.edges[k,:])            for k in range(self.Ne)] + \
            [tuple(np.flipud(self.edges[k,:])) for k in range(self.Ne)]     

        # List of vertex numbers (repeated for orderings)
        dex = list(range(self.Ne))*2 

        # lookup dictionary: identify edges by vertices
        D = dict(zip(L,dex))

        # Lengths of reference edges
        rlen = [1.,1.,np.sqrt(2.)]

        # local dofs corresponding to each edge on element
        mask = [[0,1],[1,2],[0,2]]

        # list of global dofs (vertex|edge|interior) for each element (row #)
        self.elems = np.zeros((self.Nt,self.Np),dtype='uint16')

        # Vertex dofs
        self.elems[:,:3] = self.t.copy() 

        # Determine length of every edge
        blen = np.sqrt(np.diff(self.xv[self.edges],1)**2+ \
                       np.diff(self.yv[self.edges],1)**2)

        # Global indices of nodes which lie on edges
        # j loops over points along an edge
        # k loops over edges in the mesh
        edex = [[self.Nv+j+(p-1)*k for j in range(p-1)] 
                 for k in range(self.Ne)]
 
        # Identity every boundary edge and vertex
        el,ed = np.where(self.nbr==-1) 
        bvert = zip(self.t[el,ed],self.t[el,(ed+1)%3])

        # Indices boundary edges
        bedge = tuple({D[bv] for bv in bvert})

        # Indices of boundary vertices
        bvert = tuple({b for bv in bvert for b in bv})

        # Generate Edge and Interior nodes if any
    
        if p>1:                        # Edge nodes exist

            # Indices of nodes along edge k=0,1,2 
            de = lambda k: slice(k*p+3-k,(k+1)*p+2-k)
                  
            XY = lambda j,k: np.array((self.xv[self.t[j,k%3]],
                                       self.yv[self.t[j,k%3]]))
                         
            xye = np.vstack([edgeMap(xy[de(k)],xy[k%3,:],xy[(k+1)%3,:],
                             XY(j,k%3),XY(j,(k+1)%3)) for k in range(3) 
                             for j in range(self.Nt)])
 
            self.xe = xye[:,0]
            self.ye = xye[:,1] 

            for j in range(self.Nt):   # Loop over elements
                for k in range(3):     # Loop over edges
                    inded = D[tuple(self.t[j,mask[k]])]
                    self.elems[j,slice(3+k*(p-1),3+(1+k)*(p-1))] = edex[inded]
                      
            # Identify which edge points lie on boundaries from bedge
            bedex = [self.Nv+j+(p-1)*k for j in range(p-1) for k in bedge]

            plt.plot(self.xe,self.ye,'bo')
 
            if p>2:                    # Interior nodes exist
                di  = slice(3*p,(p+2)*(p+1)/2)
                 
                XYj = lambda j: np.vstack([XY(j,k) for k in range(3)])
                
                xyi = np.vstack([interiorMap(xy[di,:],XYj(j)) 
                                 for j in range(self.Nt)])

                self.xi = xyi[:,0]
                self.yi = xyi[:,1]

                # Total number of nodes which lie on elemental boundaries
                Nb = self.Nv+self.Ne*(p-1)
                for j in range(self.Nt):
                    
                     idex = [Nb+j*self.Npi+k for k in range(self.Npi)]
                     self.elems[j,slice(3*p,self.Np)] = idex


                plt.plot(self.xi,self.yi,'go')

            else:                      # Edge nodes, but not interior nodes
                self.xi = []  
                self.yi = []  
        else:                          # No edge nodes or interior nodes
            self.xi = []  
            self.yi = []  
            self.xe = []
            self.ye = []

        self.Nep = len(self.xe)        # Number of edge points 3*self.Ne*(p-1)
        self.Ni = len(self.xi)         # Number of interior points

        
       

                 

#        np.set_printoptions(linewidth=999)
#        print(self.elems)
#        plt.show()   
       

if __name__ == '__main__':
   # number of grid points per dimension
    n = int(sys.argv[1]) 
    p = int(sys.argv[2])

    # Create tensor product grid of x,y coordinates and column stack them as vectors
    q = np.linspace(-1,1,n)
    xv,yv = [s.flatten() for s in np.meshgrid(q,q)] 

    t  = tri.Triangulation(xv,yv)

    A = assembler(t,p)



    p = int(sys.argv[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x,y = nodes(p)
    labels = [str(k) for k in range(len(x))]
    de1 = slice(3,p+2)           # Edge 1 reference indices 
    de2 = slice(p+2,2*p+1)       # Edge 2 reference indices 
    de3 = slice(2*p+1,3*p)       # Edge 3 reference indices 
    di  = slice(3*p,(p+2)*(p+1)/2)
#    ax.plot(x,y,'k.')
#    ax.plot(x[de1],y[de1],'ro')
#    ax.plot(x[de2],y[de2],'bo')
#    ax.plot(x[de3],y[de3],'go')
#    ax.plot(x[di],y[di],'yo')





    
#    plt.show()
