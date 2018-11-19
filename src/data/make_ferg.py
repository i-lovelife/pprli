from pathlib import Path
import glob
import cv2
import numpy as np
import h5py
from tqdm import tqdm
from src.data.util import shuffle
import click
ALL_ID = {'aia':0 ,'bonnie':1,  'jules':2,  'malcolm':3,  'mery':4,  'ray':5}
ALL_EMOTION = {'anger':0,   'disgust':1, 'fear':2,    'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}

@click.command()
@click.argument('dataset_root', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset_root, output_filepath):
    #print(dataset_root)
    #import pdb;pdb.set_trace()
    all_image_path = Path(dataset_root).glob('*/*/*.png')
    all_x = []
    all_y = []
    all_z = []
    #count = 0
    for image_path in tqdm(all_image_path):
        name = str(image_path).split('/')[-1].split('.')[0]
        if name == 'bonnie_surprise_1389':
            continue
        try:
            im = cv2.imread(str(image_path))
            im = cv2.resize(im, (64, 64), interpolation = cv2.INTER_CUBIC)
        except Exception as e:
            print(str(e))
            print(name)
        
        id = ALL_ID[name.split('_')[0]]
        emotion = ALL_EMOTION[name.split('_')[1]]
        all_x.append(im)
        all_y.append(emotion)
        all_z.append(id)
        #count += 1
        #if count >= 10:
        #    break
    all_x = np.array(all_x)
    all_y = np.array(all_y, dtype=np.int32).reshape(-1,)
    all_z = np.array(all_z, dtype=np.int32).reshape(-1,)
    all_x, all_y, all_z = shuffle((all_x, all_y, all_z))
    print(all_x.shape)
    print(all_y.shape)
    print(all_z.shape)
    with h5py.File(str(output_filepath), 'w') as f:
        f.create_dataset('x', data=all_x)
        f.create_dataset('y', data=all_y)
        f.create_dataset('z', data=all_z)
if __name__=='__main__':
    main()