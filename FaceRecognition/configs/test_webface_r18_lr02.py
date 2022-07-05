from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r18"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 0.1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.2
config.verbose = 2000
config.dali = False

config.rec = "/home/user/Data/CASIA-WebFace-112x112-bin"
#config.num_classes = 10572 #12*881
#config.num_classes = [12,881]
config.num_classes = [881,12]
config.num_image = 452960
config.num_epoch = 25
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30","cplfw","calfw","cfp_ff"]


### MagFace 
# config.margin_am = 0.5
# config.scale = 64
# config.l_a = 10
# config.u_a = 110
# config.l_margin = 0.45
# config.u_margin = 0.8
# config.lamda = 20


#ArcFace-v1:
config.margin = 0.5
config.scale = 64


#ArcFace-v2:
# config.margin_arc= 0.5
# config.margin_am = 0
# config.scale = 64

# # circle loss

# config.gamma = 64