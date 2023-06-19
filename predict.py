import argparse
import json
import utils2
import utils1

# Basic Usage: Predict flower name from an image with along with the probability of that name.
# python predict.py /path/to/image checkpoint

# Add our needed arguments to the argument parser to allow user to write his command easily,
# and gived each argument a description to help with the meaning of the argument.
parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint', help='Given checkpoint of a network')
parser.add_argument('--top_k', help='Return top k most likely classes')
parser.add_argument('--category_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', help='Use GPU for inference', action='store_true')


args = parser.parse_args()

# Set default values so the program doesn't crash if the user didn't enter the required arguments 
# with the exception being the data it self of course.
top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names
gpu = False if args.gpu is None else True

model = utils2.load_model(args.checkpoint)
print(model)
probs, predict_classes = utils2.predict(utils1.process_image(args.image_path), model, top_k)


# Here we attach the labels to the pictures.
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
classes = []   
for predict_class in predict_classes:
    classes.append(cat_to_name[predict_class])

print(probs)
print(classes)