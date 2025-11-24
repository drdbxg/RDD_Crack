# check_installation.py
import sys
import os

sys.path.insert(0, '/media/tongxinxi/KESU/whj/mmsegmentation')

try:
    import mmseg

    print(f"✅ MMSegmentation 版本: {mmseg.__version__}")

    # 检查关键组件
    from mmseg.registry import MODELS, DATA_PREPROCESSORS

    print("✅ 注册表导入成功")

    # 检查已注册的组件
    print("已注册的模型:", [k for k in MODELS.module_dict.keys() if 'Encoder' in k or 'Segment' in k][:5])
    print("已注册的数据预处理器:", list(DATA_PREPROCESSORS.module_dict.keys())[:5])

except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback

    traceback.print_exc()