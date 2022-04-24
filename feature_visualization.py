from pytorchyolo.models import load_model
from torchsummary import summary
import torch
import os
from pathlib import Path
from glob import glob
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import subprocess
def yolo_visualization():
    model=load_model('config/yolov3.cfg','weights/yolov3.weights')
    model.eval()
    img=cv2.resize(cv2.imread('dog.jpg'),(416,416))
    input_t=torch.tensor(img/255,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    layer_outputs = []
    def feature_sum(feature):
        return torch.sum(feature,0)
    with torch.no_grad():
        outputs,layer_outputs = model(input_t)
    os.makedirs('visualization',exist_ok=True)
    for i,layer_out in enumerate(layer_outputs):
        feature=layer_out.squeeze(0)
        if feature.dim()==3:
            feature=feature_sum(feature)
        plt.imshow(feature.cpu())
        plt.savefig(f'visualization/{i}.png')
def img_togif():
    img_l=glob('visualization/*')
    img_l=sorted(img_l,key=lambda x:int(Path(x).stem))
    f=open('2.txt', 'w')
    for item in img_l:
        item=item.replace('\\','/')
        f.write('file \''+item+'\'\n')
img_togif()
cmd="ffmpeg -r 3 -f concat  -safe 0 -i 2.txt -filter_complex split[v1][v2];[v1]palettegen[pal];[v2][pal]paletteuse=dither=sierra2_4a 11.gif -y"
subprocess.run(cmd)