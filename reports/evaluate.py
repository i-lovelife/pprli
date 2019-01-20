import sys
import ujson as json
from keras import backend as K
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
sys.path.append('../')
from src import EXPERIMENT_ROOT
from src.privater import Privater
from src.data.dataset import Ferg
from src.util.visualize import visualize
from src.util.tee_logging import TeeLogger
from src.evaluater import Ssim, Ndm, UtilityTaskEvaluater

sys.stdout = TeeLogger("stdout.log", sys.stdout)
sys.stderr = TeeLogger("stdout.log", sys.stderr)
ferg = Ferg.from_hdf5()

iter_num = 20000

vae_root = EXPERIMENT_ROOT / 'vae'
cvae_root = EXPERIMENT_ROOT / 'cvae'
ad_cvae_root = EXPERIMENT_ROOT / 'ad_cvae'
result={}
for root in [cvae_root,ad_cvae_root,vae_root]:
    config_path = root / 'config.json'
    f = config_path.open('r')
    config = json.load(f)
    privater_config = config.get('privater',{})
    privater = Privater.from_hp(privater_config)
    privater.load_weights(root / 'model_weight_30.hdf5')
    #ssim_evaluater = Ssim(verbose=True, epochs=10)
    #loss = ssim_evaluater.evaluate(ferg, privater)
    #ndm_evaluater = Ndm(verbose=True)
    #loss = ndm_evaluater.evaluate(ferg, privater)
    utility_evaluater = UtilityTaskEvaluater(verbose=True, epochs=10)
    loss = utility_evaluater.evaluate(ferg, privater)
    result[root.name] = loss
    #visualize(predicted_data['x'], np.argmax(predicted_data['p'], axis=-1), name=root.name, num_colors=predicted_data['p'].shape[-1])
    
for key,value in result.items():
    print(f'{key} : {value}')