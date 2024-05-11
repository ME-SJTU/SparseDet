point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=[
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
])
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(use_lidar=True, use_camera=True)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
backend_args = None
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles', to_float32=True,
        backend_args=None),
    dict(
        type='CustomLoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_cam3d=True,
        with_labels_cam3d=True,
        with_bbox_depth=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=False,
        training=True),
    dict(
        type='CustomPack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_cam3d',
            'gt_labels_cam3d', 'centers_2d', 'depths'
        ])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles', to_float32=True,
        backend_args=None),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(type='CustomPack3DDetInputs', keys=['img'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=None),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        test_mode=True,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=True,
                backend_args=None),
            dict(
                type='CustomLoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_attr_label=True,
                with_bbox_cam3d=True,
                with_labels_cam3d=True,
                with_bbox_depth=True),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='GlobalRotScaleTransImage',
                rot_range=[-0.3925, 0.3925],
                translation_std=[0, 0, 0],
                scale_ratio_range=[0.95, 1.05],
                reverse_angle=False,
                training=True),
            dict(
                type='CustomPack3DDetInputs',
                keys=[
                    'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
                    'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes_cam3d',
                    'gt_labels_cam3d', 'centers_2d', 'depths'
                ])
        ],
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=False,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            img='',
            sweeps='sweeps/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        box_type_3d='LiDAR',
        backend_args=None,
        use_valid_flag=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=True,
                backend_args=None),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(type='CustomPack3DDetInputs', keys=['img'])
        ],
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
        modality=dict(use_lidar=True, use_camera=True),
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            img='',
            sweeps='sweeps/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=None,
        use_valid_flag=True))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles',
                to_float32=True,
                backend_args=None),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(type='CustomPack3DDetInputs', keys=['img'])
        ],
        metainfo=dict(classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            img='',
            sweeps='sweeps/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        box_type_3d='LiDAR',
        backend_args=None,
        use_valid_flag=True))
val_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='NuScenesMetric',
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=None)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
# load_from = 'checkpoints/fcos3d_vovnet_imgbackbone-remapped.pth'
resume = True
resume_from = 'work_dirs/sparse_petr_8epoch_vovnet_gridmask_p4_800x320-nus/epoch_20.pth'
lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.3333333333333333,
        begin=0,
        end=500,
        by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=24, by_epoch=True)
]
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=2)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False, base_batch_size=32)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.SparsePETRV1.sparsepetr'])
randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
model = dict(
    type='SparsePETR',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4', 'stage5')),
    img_neck=dict(
        type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    img_roi_head=dict(
        type='SparseHead',
        num_classes=10,
        in_channels=256,
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='mmdet.L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))),
    pts_bbox_head=dict(
        type='SparsePETRHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        sample_rate=0.1,
        LID=True,
        with_position=True,
        with_dn=True,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRDNTransformer',
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
                            dropout_layer=dict(type='Dropout', drop_prob=0.1))
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
db_sampler = dict(
    data_root='data/nuscenes/',
    info_path='data/nuscenes/nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=None),
    backend_args=None)
ida_aug_conf = dict(
    resize_lim=(0.47, 0.625),
    final_dim=(320, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
num_epochs = 24
find_unused_parameters = True
launcher = 'pytorch'
work_dir = './work_dirs/sparse_petr_8epoch_vovnet_gridmask_p4_800x320-nus'
