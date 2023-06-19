import argparse
from utils1 import load_data
import utils2

# Add our needed arguments to the argument parser to allow user to write his command easily,
# and gived each argument a description to help with the meaning of the argument.
parser = argparse.ArgumentParser(description='Train a neural network on a dataset')
parser.add_argument('data_directory', help='Path to dataset')
parser.add_argument('--save_dir', help='The path to save the checkpoint')
parser.add_argument('--arch', help='architecture (default \'vgg16\')')
parser.add_argument('--learning_rate', help='Set Learning Rate')
parser.add_argument('--hidden_units', help='Hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU for training', action='store_true')


args = parser.parse_args()

# Set default values so the program doesn't crash if the user didn't enter the required arguments 
# with the exception being the data it self of course.
save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.0025 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True

# split the data
train_data, trainloader, validloader, testloader = load_data(args.data_directory)

# call the build network function to create our neural network and set its architecture and call all the needed functions
# from our organized function utility files. 
model = utils2.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx
model, criterion = utils2.train_network(model, epochs, learning_rate, trainloader, validloader, gpu)
utils2.evaluate_model(model, testloader, criterion, gpu)
utils2.save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)