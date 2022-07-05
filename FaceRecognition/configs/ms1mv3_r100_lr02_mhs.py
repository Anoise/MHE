from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.001
config.verbose = 2000
config.dali = False

config.rec = "/harbordata/ms1m-retinaface-t1"
config.num_classes = [13, 7187]

config.num_image = 5179510
config.num_epoch = 5
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30","cplfw","calfw","cfp_ff"]

config.margin = 0.5
config.scale = 64
