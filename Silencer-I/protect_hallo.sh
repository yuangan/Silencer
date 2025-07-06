
# CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.mode='refnet' attack.g_mode='+'

##### minitimestep=0
###  CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-'

##### minitimestep=0 not used anymore
### CUDA_VISIBLE_DEVICES=1 python protect/protect_hallo.py attack.img_path='th1kh/th1kh_imgs_100' attack.output_path='protect/out_th1kh_512/' attack.mode='hallo' attack.g_mode='-'

# CUDA_VISIBLE_DEVICES=0 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=100 &

# CUDA_VISIBLE_DEVICES=1 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=300 &

# CUDA_VISIBLE_DEVICES=2 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=500 &

# CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=700 &

# wait

# CUDA_VISIBLE_DEVICES=0 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=0 &

# CUDA_VISIBLE_DEVICES=1 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=100 &

# CUDA_VISIBLE_DEVICES=2 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=300 &

# CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=400 &

# wait

# CUDA_VISIBLE_DEVICES=0 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=500 &

# CUDA_VISIBLE_DEVICES=1 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=600 &

# CUDA_VISIBLE_DEVICES=2 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=700 &

# CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=800 &

# wait

# CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=900

### here is used in paper
# CUDA_VISIBLE_DEVICES=3 python protect/protect_hallo.py attack.img_path='th1kh/th1kh_imgs_100' attack.output_path='protect/out_th1kh_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=200

# CUDA_VISIBLE_DEVICES=2 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=200 &

### eps=8
# CUDA_VISIBLE_DEVICES=2 python protect/protect_hallo.py attack.img_path='celebahq_512_dataset/celebahq_512' attack.output_path='protect/out_celebahq_512/' attack.mode='hallo' attack.g_mode='-' attack.min_timesteps=200 attack.epsilon=8
