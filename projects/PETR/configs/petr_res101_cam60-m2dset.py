_base_ = [
    '../../../configs/_base_/datasets/m2dset.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.PETR.petr'])

randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-20.2, -10.2, -4.0, 110.2, 10.2, 2.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
	'car', 'bus', 'truck', 'motorbike', 'bicycle', 'person', 'child', 
	'barrier', 'trafficcone'
]
metainfo = dict(classes=class_names)

data_prefix=dict(
            pts='lidar',
            CAM_BACK_LEFT='images/image_0',
            #CAM_FRONT_30='images/image_1',
            CAM_FRONT_60='images/image_2',
            # CAM_FRONT_120='images/image_3',
            CAM_BACK_RIGHT='images/image_4',
            CAM_FRONT_RIGHT='images/image_5',
            CAM_BACK_120='images/image_6',
            CAM_FRONT_LEFT='images/image_7')

input_modality = dict(use_camera=True)
model = dict(
    type='PETR',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=9,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-30.2, -20.2, -10.0, 120.2, 20.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-30.2, -20.2, -10.0, 120.2, 20.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=9),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
	code_size=8),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=point_cloud_range))))

dataset_type = 'M2DSetDataset'
data_root = 'data/m2dset/'
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'm2dset_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            bus=5,
            truck=5,
            motorbike=5,
            bicycle=5,
            person=5,
            child=5,
            barier=5,
            trafficcone=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
            bus=4,
            truck=3,
            motorbike=6,
            bicycle=6,
            person=2,
            child=2,
            barier=2,
            trafficcone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    backend_args=backend_args)
ida_aug_conf = {
    'resize_lim': (0.47, 0.625),
    'final_dim': (540, 960),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 1920,
    'rand_flip': True,
}

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=False,
        training=True),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file='m2dset_infos_trainval.pkl',
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        modality=input_modality,
        # use_valid_flag=True,
        backend_args=backend_args))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file='m2dset_infos_val.pkl',
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        # use_valid_flag=True,
        backend_args=backend_args))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file='m2dset_infos_val.pkl',
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        # use_valid_flag=True,
        backend_args=backend_args))

# Different from original PETR:
# We don't use special lr for image_backbone
# This seems won't affect model performance
optim_wrapper = dict(
    # TODO Add Amp
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

num_epochs = 48

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    )
]

train_cfg = dict(max_epochs=num_epochs, val_interval=4)

find_unused_parameters = True

# pretrain_path can be found here:
# https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view
load_from = 'checkpoints/fcos3d.pth'
#load_from = None
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

