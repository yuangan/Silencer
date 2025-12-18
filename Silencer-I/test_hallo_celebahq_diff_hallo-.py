import os
import sys
from glob import glob
from tqdm import tqdm

prefix = int(sys.argv[3])

allfolders = glob(f'../Silencer-II/DiffAttack/adv_out/celebahq_adamw_iter200_1e-2_hallo1_512_19_+mse_t200_s100_fmask_10_100')
allimgs = glob(allfolders[prefix]+'/*')
baselinename = os.path.basename(allfolders[prefix])
outpath = f'./talking_output/{baselinename}/'
os.makedirs(outpath, exist_ok=True)

part = int(sys.argv[1])
device = int(sys.argv[2])

ll = len(allimgs) //4 + 1
print(part, ll)


for f in tqdm(allimgs[ll*part: ll*(part+1)], desc=f'test part: {part}'):
    source_image = f
    name = os.path.basename(f)[:-4]
    driving_wav = './examples/driving_audios/1.wav'
    out = os.path.join(outpath, name+'.mp4')

    if os.path.exists(out):
        print(f'exists {out}')
        continue
    
    os.system(f'CUDA_VISIBLE_DEVICES={device} python scripts/inference.py --source_image {source_image} --driving_audio {driving_wav} --output {out}')
