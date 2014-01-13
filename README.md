The hpfem2d repository contains Python scripts for implementing
an hp finite element in 2D. As of Mon Jan 13 14:40:55 MST 2014,
the included scripts are


[orthopoly.py]
A collection of routines for generating 1D orthogonal polynomials
and quadrature rules

[pkdo.py]
Evaluates the Proriol-Koornwinder-Dubiner-Owens polynomials on a 
specified 2D grid

[nodes.py] 
Generates a set of nodes on the triangle which have good 
interpolation properties

[triquad.py]
Computes a 2D Gauss quadrature rule on the triangle using 
Gauss-Jacobi quadrature and a Duffy transformation

