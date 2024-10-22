import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from load_csv import load
from show_data import show
from linear_utils import gradient_descent, model, coef_determination


def normalize_matrices(df):
    x = np.array(df.iloc[:, 0].values).reshape(-1, 1)
    Y = np.array(df.iloc[:, 1].values).reshape(-1, 1)
    nx = np.zeros(len(x), dtype=float).reshape(-1, 1)

    xmin = np.min(x)
    xmax = np.max(x)
    for i in range(len(x)):
        nx[i] = (x[i] - xmin) / (xmax - xmin)
    X = np.hstack((nx, np.ones(nx.shape)))

    return x, Y, X

def train_model(X, Y):
    learning_rate = 0.5
    n_iterations = 300

    theta = np.random.randn(2, 1)
    theta_final, cost_history = gradient_descent(X, Y, theta, learning_rate, n_iterations)
    thetas = {
        'Theta0': theta_final[0, 0].item(),
        'Theta1': theta_final[1, 0].item()
    }
    with open('thetas.json', 'w') as file:
        json.dump(thetas, file)

    predictions = model(X, theta_final)
    coef = coef_determination(Y, predictions)

    return coef, predictions, cost_history

def main():
    try:
        df = load("data.csv")
        if df is None:
            return
        if (df < 0).any().any():
            raise ValueError("Mileage and Price inputs cannot be negative.")
        
        x, Y, X = normalize_matrices(df)
        coef, predictions, cost_history = train_model(X, Y)

        print(f"The precision of the algorithm is of: {coef * 100:.2f}%")
        show(x, Y, predictions)
        # si on veut voir la courbe de coût sur le nb d iteration:
        # plt.plot(range(300), cost_history)
        # plt.show()

    except KeyboardInterrupt:
        print(" Keyboard interrupt detected.")
        return

    except Exception as e:
        print(type(e).__name__ + ":", e)
        return


if __name__ == "__main__":
    main()