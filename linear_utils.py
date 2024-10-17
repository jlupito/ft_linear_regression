import numpy as np


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