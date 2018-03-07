import matplotlib.pyplot as plt


def show_costs(costs):
    plt.plot(costs)
    plt.show()


def show_reconstruction(x, y):
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(y.reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.show()


def ask_user(question):
    ans = input(question)
    return ans and ans[0] is ('n' or 'N')
