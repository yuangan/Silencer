import os
import sys
from glob import glob
from tqdm import tqdm

prefix = int(sys.argv[2])

allfolders = glob(f'./protect/out_th1kh_512_ttt2/*_200')
allimgs = glob(allfolders[prefix]+'/th1kh_imgs_100/*')
baselinename = os.path.basename(allfolders[prefix])
outpath = f'./talking_output/th1kh_base_{baselinename}_ttt2/'
os.makedirs(outpath, exist_ok=True)
print('outpath: ', outpath)

part = int(sys.argv[1])
device = part

ll = len(allimgs) //4 + 1
print(part, ll)


for f in tqdm(allimgs[ll*part: ll*(part+1)], desc=f'test part: {part}'):
    source_image = f
    name = f.split('/')[-1][:-4]
    driving_wav = f'/home/gy/code/talking-head/hallo/th1kh/th1kh_100/audios/{name}.wav'
    out = os.path.join(outpath, 'hallo_'+name+'.mp4')

    if os.path.exists(out):
        print(f'exists {out}')
        breakpoint()
    os.system(f'CUDA_VISIBLE_DEVICES={device} python scripts/inference.py --source_image {source_image} --driving_audio {driving_wav} --output {out}')


