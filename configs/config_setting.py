from torchvision import transforms
from utils import *

from datetime import datetime

class setting_config: #训练配置中枢。采用累变量形式存储
    """
    the config of training setting.
    """

    network = 'vmunet' #使用VM-Unet架构
    model_config = {
        'num_classes': 1,   #二分类任务
        'input_channels': 3, # RGB输入 vb8i88                      8888889          888i888888888
        # ----- VM-UNet ----- #
        'depths': [2,2,2,2],  #编码器各阶段块数
        'depths_decoder': [2,2,2,1], #解码器各阶段块数
        'drop_path_rate': 0.2,  #随机深度衰减（防止过拟合）  在训练过程中随机 “丢弃” 网络中的部分路径（即整个残差块 / 子模块）
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth', #预训练权重
        'use_lsb': True,  # 新增配置
        'lsb_reduction': 4,  # 新增配置
        'use_parallel_eab': True,  # 新增：控制是否使用并行EAB
    }

    datasets = 'isic17'
    if datasets == 'isic18':
        data_path = './data/isic2018/' #数据集存放路径
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')


    criterion = BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 16
    epochs = 300   #训练轮次

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'  #带时间戳的输出目录

    print_interval = 20 # 每20个batch打印一次训练日志
    val_interval = 30  # 每30个epoch验证一次
    save_interval = 100   # 每100个epoch保存一次模型
    threshold = 0.38  # 二分类的阈值（预测概率>0.38视为正例）

    only_test_and_save_figs = False   # 是否仅测试并保存结果（不训练）
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'  # 最佳模型权重路径（测试时使用）
    img_save_path = 'PATH_TO_SAVE_IMAGES' # 测试结果图像保存路径

    #数据预处理流水线
    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True), #自定义标准化
        myToTensor(),                      #转为pytorch张量
        myRandomHorizontalFlip(p=0.5),     #50%概率水平翻转
        myRandomVerticalFlip(p=0.5),       #50%概率垂直翻转
        myRandomRotation(p=0.5, degree=[0, 360]),  #随机旋转
        myResize(input_size_h, input_size_w)
    ])
    #验证集处理
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),  #固定均值和方差
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])


#优化器是模型 “学习的方式”，决定了参数如何根据梯度调整，直接影响训练速度、收敛效果和最终性能。
    opt = 'AdamW'  #从9种优化器中选取
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if opt == 'Adadelta': #自适应学习率优化器 适用场景：文本分类、小数据集训练（无需手动调学习率）。
        lr = 0.01 # 基础学习率（默认 1.0，代码中调小为 0.01），最终学习率会基于梯度自适应缩放。
        rho = 0.9 # 梯度平方和的衰减系数（类似动量），控制历史梯度的权重（rho 越大，越依赖历史梯度）。
        eps = 1e-6 # 数值稳定项，避免分母为 0（计算时梯度平方和 + eps，防止除法出错）。
        weight_decay = 0.05 # L2 正则化系数（权重衰减），通过惩罚大权重防止模型过拟合。

    elif opt == 'Adagrad': #自适应学习率优化器  适合稀疏数据，但学习率会持续衰减（训练后期可能趋近于 0，导致模型停止学习）。
        lr = 0.01 # 初始学习率（默认 0.01），会随训练过程逐渐衰减。
        lr_decay = 0 # 学习率衰减系数（0 表示不额外衰减，默认 0）。
        eps = 1e-10 # 数值稳定项，避免分母为 0（比 Adadelta 的 eps 更小，适配稀疏数据的梯度特性）。
        weight_decay = 0.05 # L2 正则化系数，防止过拟合。

    elif opt == 'Adam': #（自适应动量优化器） 适用场景：几乎所有场景（图像分类、分割、NLP 等），代码中默认选中的优化器之一，通用性极强。
        lr = 0.001 # 初始学习率（默认 1e-3，深度学习常用默认值）。
        betas = (0.9, 0.999) # 动量参数，分别控制 “梯度的一阶矩（均值）” 和 “梯度的二阶矩（方差）” 的衰减系数：
        eps = 1e-8 # 数值稳定项，避免分母为 0。
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty)
        amsgrad = False # 是否启用 AMSGrad 变体（解决 Adam 可能出现的 “梯度方差估计偏差”，默认关闭，一般场景无需开启）。

    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum


    #lr：控制参数更新的步长  （lr 太大→训练震荡不收敛；lr 太小→收敛过慢）。
    #weight-decay：L2 正则化系数：通过惩罚模型的大权重，防止模型过度拟合训练数据（值越大，正则越强）。

    sch = 'CosineAnnealingLR'  #学习率调度器

    if sch == 'StepLR': #固定步长衰减，每隔 step_size 个 epoch，学习率乘以 gamma 衰减（固定步长、固定比例衰减）。
        step_size = epochs // 5 #衰减步长（代码中设为总 epoch 的 1/5，比如总 epoch=300，则每 60 个 epoch 衰减一次）；
        gamma = 0.5 # 衰减因子（学习率变为原来的 50%，如初始 lr=0.001，60epoch 后→0.0005，120epoch 后→0.00025）；
        last_epoch = -1 # 起始 epoch（-1 表示从 0 开始，若恢复训练可设为上次结束的 epoch）。

    elif sch == 'MultiStepLR': #多步衰减，在预设的 milestones（多个关键 epoch）处，学习率乘以 gamma 衰减（按需设置衰减节点，比 StepLR 灵活）。
        milestones = [60, 120, 150] # 衰减节点（在第 60、120、150epoch 时分别衰减）；
        gamma = 0.1 # 衰减因子（每次衰减为原来的 10%，如初始 lr=0.001→60epoch 后 0.0001→120epoch 后 0.00001）；
        last_epoch = -1 # – The index of last epoch. Default: -1.

    elif sch == 'ExponentialLR': #指数衰减，每个 epoch 的学习率都乘以 gamma，呈指数级衰减（衰减速度越来越慢）。
        gamma = 0.99 #  衰减因子（每个 epoch 学习率变为原来的 99%，如初始 lr=0.001→100epoch 后≈0.000366）；
        last_epoch = -1 # – The index of last epoch. Default: -1.

    elif sch == 'CosineAnnealingLR':  #余弦退火 学习率随 epoch 呈 余弦函数 变化 —— 从初始 lr 下降到最小 lr（eta_min），再回到初始 lr（或保持最小 lr），周期为 T_max（半周期）。
        T_max = 50 # 余弦周期的一半（即从初始 lr 下降到最小 lr 的 epoch 数，如 50epoch 完成一次 “下降”）；
        eta_min = 0.00001 # 最小学习率（下降的下限，避免 lr 趋近于 0 导致停止学习）；
        last_epoch = -1 # – The index of last epoch. Default: -1.

    elif sch == 'ReduceLROnPlateau': #基于验证集指标（如损失）动态调整—— 当指标连续 patience 个 epoch 没有改善时，才降低学习率（不是按固定 epoch 衰减，更智能）。
        mode = 'min' # 监控指标的模式（min表示指标越小越好，如损失；max表示指标越大越好，如准确率）；
        factor = 0.1 # 衰减因子（学习率变为原来的 10%）；
        patience = 10 # 耐心值（连续 10 个 epoch 指标无改善则衰减）
        threshold = 0.0001 # 指标改善的阈值（只有指标变化超过 0.0001，才视为 “有改善”，避免微小波动误判）；
        threshold_mode = 'rel' # 阈值模式（rel表示相对变化，abs表示绝对变化；如min+rel模式下，新指标需＜最佳指标 ×(1-threshold) 才视为改善）；
        cooldown = 0 # 冷却期（衰减后暂停多少个 epoch 再重新监控，避免频繁衰减）；
        min_lr = 0 # 学习率下限（默认 0，可设为 1e-6 避免 lr 过小）；
        eps = 1e-08 #数值稳定项（避免 lr 变化过小被忽略）。

    elif sch == 'CosineAnnealingWarmRestarts':  #在 CosineAnnealingLR 的基础上增加 “热重启”—— 每 T_0 个 epoch 后，学习率从最小 lr 恢复到初始 lr，然后继续余弦衰减；且每次重启后的周期 T_i 会乘以 T_mult（逐渐变长）。
        T_0 = 50 # 第一次重启的周期（前 50epoch 完成一次余弦衰减 + 重启）；
        T_mult = 2 # 周期倍增因子（第二次重启周期 = 50×2=100，第三次 = 100×2=200，以此类推）；
        eta_min = 1e-6 # 最小学习率；
        last_epoch = -1 # 起始 epoch。

    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20

# 在深度学习训练中，固定学习率存在明显缺陷：
# 初始学习率太大 → 训练震荡、不收敛；
# 初始学习率太小 → 收敛过慢、陷入局部最优；
# 训练后期学习率不变 → 难以逼近全局最优、过拟合。
#而 学习率调度器（Learning Rate Scheduler） 的核心作用是：在训练过程中动态调整学习率，让模型在不同阶段使用合适的学习率 —— 前期用较大学习率快速探索参数空间，后期用较小学习率精细调整参数，最终实现更快收敛、更好的泛化性能。




# setting_config
# ├── model_config（模型结构）
# ├── datasets（数据路径）
# ├── transformer（预处理）
# ├── optimizer（优化器）
# └── scheduler（学习率）