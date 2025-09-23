def fun1(x, y):
    return x + y

def fun2(x, y):
    return x - y

def fun3(x, y):
    return x * y

def fun4(x, y):
    # Combines add, subtract, and multiply of x and y, returns the total
    return fun1(x, y) + fun2(x, y) + fun3(x, y)
