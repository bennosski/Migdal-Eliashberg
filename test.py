
from numpy import *

x = random.randn(4,5,2,2)
y = random.randn(4,5,2,2)
tau = random.randn(2,2)

print(einsum('ab,xybc->xyac', tau, x)-einsum('ab,...bc->...ac', tau, x))
