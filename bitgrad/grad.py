import math

class Value:
    def __init__(self, data, _children=(), _op = None):
        self.data = data
        self._prev = set(_children) # maybe changing this to set
        self._op= _op
        self.grad = 0 #initialization
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), 'add')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), 'mul')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), 'sub')

        def _backward():
            self.grad += out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward

        return out

    def __truediv__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data**(-1.0), (self, other), 'truediv')

        def _backward():
            self.grad += other.data**(-1.0) * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x) +1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        t = math.exp(x)
        out = Value(t, (self,) , "exp")

        def _backward():
            self.grad += t * out.grad
        out._backward = _backward

        return out
    def __pow__(self, other):
        assert isinstance(other, (int,float)), " only support int or float powers for now"
        out = Value(self.data**other, (self,) , f"pow{other}")

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()






