from bitgrad.grad import Value

a = Value(-3.0)
b = Value(1.0)
c = a + b
d = a**2 + a * b
c += c + 1
d += d * 2 + (b + a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 220.5227, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints -356.9633, the numerical value of dg/da
print(f'{b.grad:.4f}') # prints -230.9762, the numerical value of dg/db