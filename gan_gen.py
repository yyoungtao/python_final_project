import keras
from genetic import Genetic

def train_gan(gans, dataset):
    for gan in gans:
        gan.train(dataset)

def get_average_accuracy(gans):
    return 0

def generate(generations, population, nn_param_chices, dataset):
    genetic = Genetic(nn_param_chices)
    gans = genetic.create_population(population)

    for i in range(generations):
        train_gan(gans, dataset)

        average_accuracy = get_average_accuracy(gans)

        if i != generations - 1:
            gans = genetic.evolve(gans)

    gans = sorted(gans, key=lambda x: x.accuracy, reverse=True)

    #print networks


def print_networks():
    return 0


def main():
    generations = 10
    population = 20
    dataset = 'random uniform noise'

    nn_param_choices = {
        'nb_neurons': [2, 4, 8, 16, 32, 64],
        'nb_layers':[1, 2, 3, 4],
        'activation':['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer':['rmsprop', 'adam', 'sgd', 'adagrad',
'adadelta', 'adamax', 'nadam']
    }
    generate(generations, population, nn_param_choices, dataset)


if __name__ == '__main__':
    main()