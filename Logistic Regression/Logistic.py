import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, weights=None, lr=0.01, reg_coef=0.01):
        self.lr = lr
        self.w = weights if weights else np.zeros((21,1)) # np.random.randn(20,1)  21 because we added bias too.
        self.lanbda = reg_coef # lambda

    
    def sigmoid(self, x):
        z = np.dot(x, self.w)
        return 1 / (1 + np.exp(-z))
        
    
    def cost(self, x, y):
        eps = 0.00000000001 # just to prevent that y_hat might be 0 and log(0) is nan ! 
        y_hat = self.sigmoid(x)
        y_hat += eps 
        loss =  -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + self.lanbda*np.sum((self.w)**2) # l2 Reg
        return loss

    def train(self, x, y):

        # according to wikipedia https://en.wikipedia.org/wiki/Cross-entropy
        # the derivative of cross-entropy is just  Σx(y_hat - y)
        y_hat = self.sigmoid(x)
        grads = np.dot(x.T, (y_hat - y)) + ( 2 * self.lanbda * self.w ) 
        self.w = self.w - (grads * self.lr)
        
    
    def predict(self, x):
        return self.sigmoid(x)

    def classify(self, x):
        return 1 if self.sigmoid(x) > 0.5 else 0


    def params(self):
        return self.w


df = pd.read_csv("Life Expectancy Data.csv") # Reading file
df = df.drop(columns="Country") # dropping categorical variable

status = df['Status'] # mapping target variable to 0 or 1 
target = []
for i in status:
    if i == 'Developing':
        target.append(0)
    else:
        target.append(1)

df['Status'] = target # replacing target variable with its mapping to 0 & 1
df = df.dropna() # handling missing values with droping them.
# df['constant Term'] = np.ones(df.shape[0])   # this is weird because when i add bias to our data before cleaning, it becomes NaN, in the process ! 
# TODO: Why?

df = df.sample(frac=1) # shuffling dataset
y = np.array(df['Status']).reshape(-1, 1) # take our dependant variable 

sample_status = df.iloc[1000]['Status']

df = df.drop(columns='Status')
df = (df - df.mean()) / df.std() # standardize data, because of big numbers
df['Constant Term'] = np.ones(df.shape[0])   # Adding constant to our input

test_sample = df.iloc[1000]

# when adding the constant term after data cleaning is done, its makes our loss much better !
# TODO: @Question? as Jeremy told: we always add this 1 vector to our input as bias, but shouldnt it be some other values rather than 1?
# Answer: it should be 1, because when our model finds its optimal param, that parameter will multiply by 1 and our final function has the optimal bias
# and everything works as it should! 
# @Note ToMyself: always add 1 vector to input ! this one was a good example of showing importance of bias
x = np.array(df)

 

validation_breakpoint = int(np.floor((x.shape[0] * 0.8)))
x_train = x[:validation_breakpoint]
y_train = y[:validation_breakpoint]
x_valid = x[validation_breakpoint:]
y_valid = y[validation_breakpoint:]

print(x.shape, y.shape)

# logistic = LogisticRegression(lr=0.001234)
logistic = LogisticRegression(lr=0.0001, reg_coef=0.0001)   
# logistic = LogisticRegression(lr=0.00123, reg_coef=0.00123)

# After Adding the bias term to the equations it got much better! thats crazy. from 0.5 loss to 0.15 !
# TODO: @Question? why adding just a constant term made our loss much better and our model was more accurate?

print(logistic.cost(x_train, y_train))


loss_valid = []
loss_train = []
for i in range(1000):
    loss_train.append(logistic.cost(x_train, y_train))
    loss_valid.append(logistic.cost(x_valid, y_valid))
    logistic.train(x_train, y_train)
    if (i + 1) % 100 == 0:
        print("Epoch:", i+1, "Train:", logistic.cost(x_train, y_train), "Valid: ", logistic.cost(x_valid, y_valid))

output = logistic.classify(np.array(test_sample))
print(f"Result of sigmoid of test_sample which is captured randomly each time {logistic.predict(np.array(test_sample))}, Actuacl y: {sample_status}, Predicted y: {output}")

plt.ion()
plt.figure()
plt.title("Loss Train")
plt.plot(loss_train)
plt.show()

plt.figure()
plt.title("Loss Valid")
plt.plot(loss_valid)
plt.show()


plt.pause(0)