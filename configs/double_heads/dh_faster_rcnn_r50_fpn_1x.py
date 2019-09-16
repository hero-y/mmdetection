# model settings
model = dict(
    type='DoubleHeadRCNN',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    reg_roi_scale_factor=1.3,
    bbox_head=dict(
        type='DoubleConvFCBBoxHead',
        num_convs=4,
        num_fcs=2,
        in_channels=256,
        conv_out_channels=1024,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,  #class_agnostic：类别不可只，也就是说bbox输出的时候只考虑是否为前景，后续分类再根据bbox的分数分类，一个框可以对应多个类别
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),  #忽略bbox的阈值，当gt中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='RandomSampler',
            num=256, #这256是采样的anchor数，也就是pos+neg的数量
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,  #允许在bbox周围外扩一定的像素
        pos_weight=-1,  #正样本权重，-1表示不改变原始的权重
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000, #每一层都会根据分数留2000个来nms
        nms_post=2000, #每一层在nms后都会留最多2000个
        max_num=2000, #所有层加在一起最后留2000个
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512, #一个图像的所有的proposal的最大值,如果是两个图像，roi的大小就是(1024,7*7*256)
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False, #在所有的fpn层内做nms
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # 图像初始化，减去均值除以方差
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),#将较大者/图像中h,w的较大者  将较小者/图像中h,w的较小者  选择较小的作为放大比例 ，效果就是在没有pad之前，当img的长边等于大值时，短边就会小于小值，当img的短边等于小值时，长边就会小于大值
    dict(type='RandomFlip', flip_ratio=0.5),# 左右翻转的概率,再dataset.transform中设定，随机值和flip_ratio比较，来判断概率
    dict(type='Normalize', **img_norm_cfg), 
    dict(type='Pad', size_divisor=32), # 对图像resize时的最小单位，所有图像都会被resize成32的倍数,这个即使形成的pad_shape
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2, #一个gpu的线程数，线程数越多处理的越快
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/', # 图片前缀，即图片的路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer gpu=8时 lr=0.02, gpu=4时 lr=0.01， gpu=1时 lr=0.0025
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  #梯度均衡参数
# learning policy
lr_config = dict(
    policy='step',  #优化策略
    warmup='linear',  #初始学习率增加的策略，线性增加
    warmup_iters=500, #在初始的500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3, #最开始的lr为lr/3,即0.02/3
    step=[8, 11])
checkpoint_config = dict(interval=1) #每1个epoch存储一次模型 interval间隔
# yapf:disable
log_config = dict(
    interval=1, #每50个batch输出一次
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl') # 分布式参数
log_level = 'INFO'
work_dir = './work_dirs/dh_faster_rcnn_r50_fpn_1x'
load_from = None
resume_from = None  # 恢复训练模型的路径
workflow = [('train', 1)] # 当前工作区名称
