import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PolynomialReg:
    def __init__(self, lr=0.01, deg=None, pretrained_weights=None):
        self.lr = lr           
        self.deg = deg
        self.w = pretrained_weights if pretrained_weights else np.random.rand(deg + 1, 1) # 1 for bias
        if pretrained_weights:
            if deg+1 != len(pretrained_weights):
                raise ValueError("weight and degree dosent match. weights should be +1 bigger than degree")
        

    def fix_input(self, x):
        x_new = x
        for i in range(1, self.deg):
            new_col = x ** (i+1)
            x_new = np.append(x_new, new_col.reshape(-1, 1), axis=1)
        x_new = np.append(x_new, np.ones((x_new.shape[0], 1)), axis = 1) # add ones to x for bias as last column


        # second implementation which look better
        # x_new = np.ones((len(x), 1))
        # for i in range(self.deg):
        #     new_col = x ** (i + 1)
        #     x_new = np.append(x_new, new_col, axis=1)
        # Actually it dosent matter wether to put bias at end or begin of x_matrix, you just need to remember that
        # wherever you put it, your bias is correspanding to it in your weights vector! 
        
    
        return x_new
        
    
    def cost(self, x, y):
        x_new = self.fix_input(x)
        # MSE loss function
        y_hat = np.dot(x_new, self.w)
        return np.mean((y - y_hat) ** 2)

    
    def train(self, x, y):
        x_new = self.fix_input(x)
        y_hat = np.dot(x_new, self.w)
        error = y - y_hat
        # Gradients of MSE: -2 * x * error / len (x) 
        grads = (-2/x_new.shape[0]) * (np.dot(x_new.T, error))
        self.w = self.w - (grads * self.lr)

    
    def predict(self, x):
        x_new = self.fix_input(x)
        return np.dot(x_new, self.w)

    
    def params(self):
        return self.w




df = pd.read_csv("HW_x_2d.csv")
df = df.drop(columns="x1") # get rid of wrong data in our dataset
df = df.sample(frac=1)

x = np.array(df["x2"]).reshape(-1, 1)
y = np.array(df["y"]).reshape(-1, 1)

validation_point = int((x.shape[0]*0.8))

x_train = x[:validation_point]
y_train = y[:validation_point]
x_valid = x[validation_point:]
y_valid = y[validation_point:]


# deg = 2 -> lr = 0.01
# deg = 3 -> lr = 0.001
# deg = 7 -> lr = 0.00000005 or something like that + more epochs

poly2 = PolynomialReg(lr=0.01, deg=2)
poly3 = PolynomialReg(lr=0.001, deg=3)
poly7 = PolynomialReg(lr=0.0000009321, deg=7)


loss_train = []
loss_valid = []
loss_valid3 = []
loss_valid7 = []

plt.ion()
f1 = plt.figure()
plt.scatter(x, y, alpha=0.3)

for i in range(1000):
    poly2.train(x_train, y_train)
    poly3.train(x_train, y_train)
    poly7.train(x_train, y_train)

    loss_train.append(poly2.cost(x_train, y_train))
    loss_valid.append(poly2.cost(x_valid, y_valid))
    loss_valid3.append(poly3.cost(x_valid, y_valid))
    loss_valid7.append(poly7.cost(x_valid, y_valid))
    if (i + 1) % 50 == 0:
        plt.scatter(x, poly2.predict(x), alpha=0.06, marker='_')
        print("Epoch:", i+1,"Train_Loss:", poly2.cost(x_train, y_train), "Validation_Loss:", poly2.cost(x_valid, y_valid))

plt.title("The Chart is showing Training Curves of deg=2")        
plt.show()
print("These are the weights:\n", poly2.params())

for i in range(20000):
    poly7.train(x_train, y_train)

p = poly2.predict(x_valid)
f2 = plt.figure()
plt.scatter(x_valid, y_valid)
plt.scatter(x_valid, p)
plt.title("This Chart shows how good our curve is on validation set with deg = 2")
print("our final loss in validation set:", poly2.cost(x_valid, y_valid))
plt.show()

p = poly3.predict(x_valid)
f2 = plt.figure()
plt.scatter(x_valid, y_valid)
plt.scatter(x_valid, p)
plt.title("This Chart shows how good our curve is on validation set with deg = 3")
print("our final loss in validation set:", poly3.cost(x_valid, y_valid))
plt.show()


p = poly7.predict(x_valid)
f2 = plt.figure()
plt.scatter(x_valid, y_valid)
plt.scatter(x_valid, p)
plt.title("This Chart shows how good our curve is on validation set with deg = 7")
print("our final loss in validation set:", poly7.cost(x_valid, y_valid))
plt.show()


plt.pause(0)



