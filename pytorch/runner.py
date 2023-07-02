import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from bts import BtsModel
from bts_dataloader import *

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def load_image(image_path):
    # Load image, then convert it to RGB and normalize it to [0, 1]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='resnext101_bts')#default='densenet161_bts')#default='resnext101')
#parser.add_argument('--data_path', type=str, help='path to the data', required=True)
#parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='/Users/lewishickley/BBK/MscThesis/models/initialSet/bts/pytorch/models/bts_eigen_v2_pytorch_resnext101/model')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--image_path',                type=str,   help='path to the image', default='/Users/lewishickley/Downloads/InitialTestImage.jpg') #TODO change this default and the handling around it.


args = parser.parse_args()

args.mode = 'test'
#dataloader = BtsDataLoader(args, 'test')

transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    
model = BtsModel(params=args)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()
#model.cuda() #Turn this back on one day

num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("Total number of parameters: {}".format(num_params))

img = load_image(args.image_path)
img = transform(img)
img = img.unsqueeze(0).float()

with torch.no_grad():
    output = model(img, None)

output = output[-1]

print(output.__class__)
#print(length(output))

# The output is a depth map, it's up to us how to process it
# For simplicity, let's convert it to numpy and squeeze unnecessary dimensions
output = output.cpu().numpy().squeeze()

#Return a plot of the data so we can visualise how it is doing.
#TODO Write a function to save this image.
plt.imshow(output, cmap='inferno')
plt.colorbar()
plt.show()