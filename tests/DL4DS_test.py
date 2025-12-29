import pytest
import numpy as np
import xarray as xr
import dl4ds as dds
import tensorflow as tf

def test_dds_trainer_integration():
    """
    测试 dl4ds 训练流程的冒烟测试
    """
    # 1. 构造极小的模拟数据 (代替 climetlab 下载)
    shape = (10, 8, 8, 1) # [时间, 宽, 高, 通道]
    data = np.random.rand(*shape).astype('float32')
    
    # 转换为 xarray (dl4ds 常用格式)
    da = xr.DataArray(data, dims=['time', 'lat', 'lon', 'channel'])
    
    # 2. 极简配置
    ARCH_PARAMS = dict(
        n_filters=4, # 减少滤镜
        n_blocks=2,   # 减少块
        normalization=None,
        activation='relu',
        localcon_layer=True)

    # 3. 初始化 Trainer
    trainer = dds.SupervisedTrainer(
        backbone='resnet',
        upsampling='spc', 
        data_train=da, 
        data_val=da,
        data_test=da,
        scale=2, # 缩小倍数以加快速度
        batch_size=2, 
        loss='mae',
        epochs=1,     # 只跑 1 个 epoch 验证流程
        show_plot=False, # CI 环境必须关闭
        verbose=True, 
        device='CPU', 
        **ARCH_PARAMS)

    # 4. 运行
    try:
        trainer.run()
        run_success = True
    except Exception as e:
        run_success = False
        print(f"Training failed: {e}")

    # 5. 断言
    assert run_success is True
    # 也可以检查模型是否生成
    assert trainer.model is not None