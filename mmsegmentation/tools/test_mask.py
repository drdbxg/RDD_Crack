import os
import mmcv
import numpy as np
from mmengine.registry import init_default_scope
from mmseg.apis import init_model, inference_model

# --------------------------
# 修改为你的配置与权重路径
# --------------------------
config = 'work_dirs/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py'
checkpoint = 'work_dirs/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512/iter_40000.pth'

# 初始化模型
model = init_model(config, checkpoint, device='cuda:0')
init_default_scope('mmseg')

# 保存路径
save_dir = 'result/baseline'
os.makedirs(save_dir, exist_ok=True)

# 你的 test文件夹路径
test_dir = os.path.join('data/UAVCrack/img_dir/test')

# 获取所有图像
img_list = sorted(os.listdir(test_dir))

for img_name in img_list:
    img_path = os.path.join(test_dir, img_name)

    # 推理
    result = inference_model(model, img_path)
    pred = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

    # 输出文件名 = 输入文件名 + ".png"
    out_name = img_name + '.png'
    save_path = os.path.join(save_dir, out_name)

    # 保存mask
    mmcv.imwrite(pred * 255, save_path)

print("🎉 All prediction masks saved to ./result/")
