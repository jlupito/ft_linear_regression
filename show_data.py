import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def show(x, y, predictions,):
    
    plt.scatter(x, y, marker='+', label="dataset")
    plt.plot(x, predictions, c='r', label='prediction')
    plt.title("ft_linear_regression f(x) = ax + b", pad=10)
    plt.xlabel("mileage (km)")
    plt.ylabel("price (€)")
    plt.legend()

    def format_k(value, tick_number):
        if value >= 1000:
            return f'{int(value/1000)}k'
        return int(value)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_k))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_k))

    plt.show()
