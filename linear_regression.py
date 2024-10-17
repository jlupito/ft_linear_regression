import numpy as np
import pandas as pd
from load_csv import load
from show_data import show
from sklearn import preprocessing
import matplotlib.pyplot as plt
import json


def model(X, theta):
    """
    F = X.θ (f(x)=ax+b), calculates the matricial product of X by θ.
    """
    return X.dot(theta)

def cost_function(X, y, theta):
    """
    J(θ) = (1/2m)Σ(Xθ - Y)², calculates the cost ie the mean squared error (mse).
    """
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    """
    Calculates the gradient, ie all derivatives of J for each parameter from vector θ.
    """
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    """
    Iterates n times by learning_rate step to find θ (parameters with lowest cost).
    Also returns the cost history to check the cost at each iteration.
    """
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history

def coef_determination(y, pred):
    """
    Evaluates performance of the model between 0 and 1,
    the closer to 1 the more performing.
    """
    # u = residu de la somme des carrés
    u = ((y - pred)**2).sum()
    # v = somme totale des carrés
    v = ((y - y.mean())**2).sum()
    return 1 - (u/v)

def main():
    try:
        df = load("data.csv")
        if df is None:
            return
        if (df < 0).any().any():
            raise ValueError("Mileage and Price inputs cannot be negative.")
        
        transformer = preprocessing.MinMaxScaler().fit(df[['km']])
        transformed = transformer.transform(df[['km']])
        
        x = np.array(df.iloc[:, 0].values).reshape(-1, 1)
        Y = np.array(df.iloc[:, 1].values).reshape(-1, 1)
        nx = transformed[:, 0].reshape(-1, 1)
        # print(Y.shape)
        # print(nx.shape)
        
        X = np.hstack((nx, np.ones(nx.shape)))
        # print(X)

        theta = np.random.randn(2, 1)
        learning_rate = 0.5
        n_iterations = 300
        theta_final, cost_history = gradient_descent(X, Y, theta, learning_rate, n_iterations)
        thetas = {
            'Theta0': theta_final[0, 0].item(),
            'Theta1': theta_final[1, 0].item()
        }

        with open('thetas.json', 'w') as file:
            json.dump(thetas, file)

        predictions = model(X, theta_final)
        coef = coef_determination(Y, predictions)
        print(f"The precision of the algorithm is of: {coef * 100:.2f}%")
        show(x, Y, predictions)
        # si on veut voir la courbe de coût sur le nb d iteration:
        # plt.plot(range(1000), cost_history)
        # plt.show()

    except KeyboardInterrupt:
        print(" Keyboard interrupt detected.")
        return

    except Exception as e:
        print(type(e).__name__ + ":", e)
        return


if __name__ == "__main__":
    main()