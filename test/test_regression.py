from bitgrad.nn import MLP

n = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

for itter in range(100):
    for p in n.parameters():
        p.grad = 0.0
    y_pred = [n(x) for x in xs]
    loss = 1/(len(ys))*sum((y_pred[i] - ys[i])**2 for i in range(len(ys)))
    print(f"loss = {loss}")
    loss.backward()
    for p in n.parameters():
        p.data -= p.grad * 0.1

print(y_pred) # prints [Value(data=0.933), Value(data=-0.958), Value(data=-0.950), Value(data=0.929)]
print(loss) # prints 0.004