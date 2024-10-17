import json


def price_estimation(mileage, theta0, theta1):
    km = (float(mileage) - 22899) / (240000 - 22899)
    price = theta0 + (theta1 * km)
    return round(price)


def main():
    try:
        with open('thetas.json', 'r') as file:
            thetas = json.load(file)

        theta0 = thetas['Theta1']
        theta1 = thetas['Theta0']
        # print(theta0)
        # print(theta1)

        while(1):
            mileage = input("What is the mileage of your car ? ")
            if not mileage:
                print('\nPlease provide a value.')
            else:
                try:
                    assert(all([c in '0123456789' for c in mileage]))
                    break
                except:
                    print('\nPlease provide a VALID value.')

        price = price_estimation(mileage, theta0, theta1)
        if price <= 0:
            print(f"You may as well give your car away.")
        else:
            print(f"You should sell your car at a price of {price:,} euros.")

    except KeyboardInterrupt:
        print(" Keyboard interrupt detected.")
        return

    except Exception as e:
        print(type(e).__name__ + ":", e)
        return


if __name__ == "__main__":
    main()