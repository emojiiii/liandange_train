#!/bin/bash
# LoRA train script by @Akegarasu modify by @bdsqlsz

#训练模式(Lora、db、Sdxl_lora、Sdxl_db、sdxl_cn3l、controlnet(未完成))
train_mode="sdxl_db"

# Train data path | 设置训练用模型、图片
pretrained_model="./Stable-diffusion/sd_xl_base_1.0_fixvae_fp16_V2.safetensors" # base model path | 底模路径
vae=""
is_v2_model=0 # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
v_parameterization=0 # parameterization | 参数化 v2 非512基础分辨率版本必须使用。
train_data_dir="./train/color_trace" # train dataset path | 训练数据集路径
reg_data_dir=""	# reg dataset path | 正则数据集化路径
network_weights="" # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
training_comment="this LoRA model created from bdsqlsz by bdsqlsz'script" # training_comment | 训练介绍，可以写作者名或者使用触发关键词
dataset_class=""

#差异炼丹法
base_weights="" #指定合并到底模basemodel中的模型路径，多个用空格隔开。默认为空，不使用。
base_weights_multiplier="1.0" #指定合并模型的权重，多个用空格隔开，默认为1.0。

# Train related params | 训练相关参数
resolution="1024,1024" # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
batch_size=1 # batch size 一次性训练图片批处理数量，根据显卡质量对应调高。
max_train_epoches=20 # max train epoches | 最大训练 epoch
save_every_n_epochs=5 # save every n epochs | 每 N 个 epoch 保存一次

gradient_checkpointing=1 #梯度检查，开启后可节约显存，但是速度变慢
gradient_accumulation_steps=0 # 梯度累加数量，变相放大batchsize的倍数

network_dim=64 # network dim | 常用 4~128，不是越大越好
network_alpha=1 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

#dropout | 抛出(目前和lycoris不兼容，请使用lycoris自带dropout)
network_dropout=0 # dropout 是机器学习中防止神经网络过拟合的技术，建议0.1~0.3 
scale_weight_norms=1.0 #配合 dropout 使用，最大范数约束，推荐1.0
rank_dropout=0 #lora模型独创，rank级别的dropout，推荐0.1~0.3，未测试过多
module_dropout=0 #lora模型独创，module级别的dropout(就是分层模块的)，推荐0.1~0.3，未测试过多
caption_dropout_every_n_epochs=0 #dropout caption
caption_dropout_rate=0.1 #0~1
caption_tag_dropout_rate=0.1 #0~1

train_unet_only=1 # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
train_text_encoder_only=0 # train Text Encoder only | 仅训练 文本编码器

seed=1026 # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#noise | 噪声
noise_offset=0.05 # help allow SD to gen better blacks and whites，(0-1) | 帮助SD更好分辨黑白，推荐概念0.06，画风0.1
adaptive_noise_scale=0.05 #自适应偏移调整，10%~100%的noiseoffset大小
multires_noise_iterations=0 #多分辨率噪声扩散次数，推荐6-10,0禁用。
multires_noise_discount=0 #多分辨率噪声缩放倍数，推荐0.1-0.3,上面关掉的话禁用。

#lycoris组件
enable_lycoris=0 # 开启lycoris
conv_dim=0 #卷积 dim，推荐＜32
conv_alpha=0 #卷积 alpha，推荐1或者0.3
algo="full" # algo参数，指定训练lycoris模型种类，包括lora(就是locon)、loha、IA3以及lokr、dylora、full(DreamBooth先训练然后导出lora) ，6个可选
dropout=0 #lycoris专用dropout
preset="attn-mlp" #预设训练模块配置
#full: default preset, train all the layers in the UNet and CLIP|默认设置，训练所有Unet和Clip层
#full-lin: full but skip convolutional layers|跳过卷积层
#attn-mlp: train all the transformer block.|kohya配置，训练所有transformer模块
#attn-only：only attention layer will be trained, lot of papers only do training on attn layer.|只有注意力层会被训练，很多论文只对注意力层进行训练。
#unet-transformer-only： as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.|和attn-mlp类似，但是关闭te训练
#unet-convblock-only： only ResBlock, UpSample, DownSample will be trained.|只训练卷积模块，包括res、上下采样模块
#./toml/example_lycoris.toml: 也可以直接使用外置配置文件，制定各个层和模块使用不同算法训练，需要输入位置文件路径，参考样例已添加。

factor=8 #只适用于lokr的因子，-1~8，8为全维度
block_size=4 #适用于dylora,分割块数单位，最小1也最慢。一般4、8、12、16这几个选
use_tucker=1 #适用于除 (IA)^3 和full
use_scalar=1 #根据不同算法，自动调整初始权重
train_norm=1 #归一化层

#dylora组件
enable_dylora=0 # 开启dylora，和lycoris冲突，只能开一个。
unit=4 #分割块数单位，最小1也最慢。一般4、8、12、16这几个选

#Lora_FA
enable_lora_fa=0 # 开启lora_fa，和lycoris、dylora冲突，只能开一个。

#oft
enable_oft=0 # 开启oft，和已上冲突，只能开一个。

# Learning rate | 学习率
lr="5e-6"
unet_lr="1e-4"
text_encoder_lr="2e-5"
lr_scheduler="constant_with_warmup" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 推荐默认cosine_with_restarts或者polynomial，配合输出多个epoch结果更玄学
lr_warmup_steps=50 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
lr_scheduler_num_cycles=1 # restarts nums | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值

# 优化器
optimizer_type="adaFactor" 
# 可选优化器"adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation",  
# 新增优化器"Lion8bit"(速度更快，内存消耗更少)、"DAdaptAdaGrad"、"DAdaptAdan"(北大最新算法，效果待测)、"DAdaptSGD"
# 新增DAdaptAdam、DAdaptLion、DAdaptAdanIP，强烈推荐DAdaptAdam
# 新增优化器"Sophia"(2倍速1.7倍显存)、"Prodigy"天才优化器，可自适应Dylora
# PagedAdamW8bit、PagedLion8bit、Adan、Tiger
d_coef="0.5"
d0="1e-4" #dadaptation以及prodigy初始学习率

shuffle_caption=1 # 随机打乱tokens
keep_tokens=1 # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。
prior_loss_weight=1 #正则化权重,0-1
min_snr_gamma=0 #最小信噪比伽马值，减少低step时loss值，让学习效果更好。推荐3-5，5对原模型几乎没有太多影响，3会改变最终结果。修改为0禁用。
ip_noise_gamma=0.1 #误差噪声添加，防止误差累计
debiased_estimation_loss=1 #信噪比噪声修正，minsnr高级版
weighted_captions=0 #权重打标，默认识别标签权重，语法同webui基础用法。例如(abc), [abc],(abc:1.23),但是不能在括号内加逗号，否则无法识别。一个文件最多75个tokens。

# block weights | 分层训练
enable_block_weights=0 #开启分层训练，和lycoris冲突，只能开一个。
down_lr_weight="1,0.2,1,1,0.2,1,1,0.2,1,1,1,1" #12层，需要填写12个数字，0-1.也可以使用函数写法，支持sine, cosine, linear, reverse_linear, zeros，参考写法down_lr_weight=cosine+.25 
mid_lr_weight="1"  #1层，需要填写1个数字，其他同上。
up_lr_weight="1,1,1,1,1,1,1,1,1,1,1,1"   #12层，同上上。
block_lr_zero_threshold=0  #如果分层权重不超过这个值，那么直接不训练。默认0。

enable_block_dim=0 #开启dim分层训练
block_dims="128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128" #dim分层，25层
block_alphas="16,16,32,16,32,32,64,16,16,64,64,64,16,64,16,64,32,16,16,64,16,16,16,64,16"  #alpha分层，25层
conv_block_dims="32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" #convdim分层，25层
conv_block_alphas="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" #convalpha分层，25层

# block lr
enable_block_lr=0
block_lr="0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,0"

# Output settings | 输出设置
output_name="16GDB" # output model name | 模型保存名称
save_model_as="safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors
mixed_precision="bf16" # bf16效果更好，默认fp16
save_precision="bf16" # bf16效果更好，默认fp16
full_fp16=0 #开启全fp16模式，自动混合精度变为fp16，更节约显存，实验性功能
full_bf16=1 #选择全bf16训练，必须30系以上显卡。

# Resume training state | 恢复训练设置
save_state=0 # save training state | 保存训练状态 名称类似于 <output_name>-??????-state ?????? 表示 epoch 数
resume="" # resume from state | 从某个状态文件夹中恢复训练 需配合上方参数同时使用 由于规范文件限制 epoch 数和全局步数不会保存 即使恢复时它们也从 1 开始 与 network_weights 的具体实现操作并不一致

#保存toml文件
output_config=0 #开启后直接输出一个toml配置文件，但是无法同时训练，需要关闭才能正常训练。
config_file="./toml/$output_name.toml" #输出文件保存目录和文件名称，默认用模型保存同名。

#输出采样图片
enable_sample=0 #1开启出图，0禁用
sample_every_n_epochs=5 #每n个epoch出一次图
sample_prompts="./toml/1girl.txt" #prompt文件路径
sample_sampler="euler_a" #采样器 'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'

#wandb 日志同步
wandb_api_key="" # wandbAPI KEY，用于登录

# 其他设置
enable_bucket=1 #开启分桶
min_bucket_reso=512 # arb min resolution | arb 最小分辨率
max_bucket_reso=1536 # arb max resolution | arb 最大分辨率
persistent_workers=1 # makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage | 跑的更快，吃内存。大概能提速2倍
vae_batch_size=4 #vae批处理大小，2-4
clip_skip=2 # clip skip | 玄学 一般用 2
cache_latents=1 #缓存潜变量
cache_latents_to_disk=1 # 缓存图片存盘，下次训练不需要重新缓存，1开启0禁用
sdpa=0 #设置为0使用torch2.0自带的优化sdpa，否则使用xformers

#SDXL专用参数
#https://www.bilibili.com/video/BV1tk4y137fo/
min_timestep=0 #最小时序，默认值0
max_timestep=1000 #最大时序，默认值1000
cache_text_encoder_outputs=1 #开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
cache_text_encoder_outputs_to_disk=1 #开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
no_half_vae=0 #禁止半精度，防止黑图。无法和mixed_precision混合精度共用。
bucket_reso_steps=32 #SDXL分桶可以选择32或者64。32更精细分桶。默认为64

#db checkpoint train
stop_text_encoder_training=0
no_token_padding=0 #不进行分词器填充
learning_rate_te="5e-8"

#sdxl_db
diffusers_xformers=0
train_text_encoder=1
learning_rate_te1="5e-8"
learning_rate_te2="5e-8"

#sdxk_cn3l
conditioning_data_dir=""
cond_emb_dim=32

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
source ./venv/bin/activate

export HF_HOME="huggingface"
export XFORMERS_FORCE_DISABLE_TRITON="1"
network_module="networks.lora"
ext_args=()

launch_script="train_network"

if [[ ! "$train_mode" =~ "lora" && "$train_mode" != "sdxl_cn3l" ]]; then
  network_module=""
  network_dim=""
  network_alpha=""
  conv_dim=""
  conv_alpha=""
  network_weights=""
  enable_block_weights=0
  enable_block_dim=0
  enable_lycoris=0
  enable_dylora=0
  enable_lora_fa=0
  enable_oft=0
  unet_lr=""
  text_encoder_lr=""
  train_unet_only=0
  train_text_encoder_only=0
  training_comment=""
  prior_loss_weight=1
  network_dropout="0"
fi

if [[ "$train_mode" =~ "db" ]]; then
  if [ "$train_mode" = "db" ]; then
    laungh_script="train_db"
    if [ "$no_token_padding" -ne 0 ]; then
      ext_args+=("--no_token_padding")
    fi
    if [ "$stop_text_encoder_training" -ne 0 ]; then
      if [ "$gradient_accumulation_steps" -ne 0 ]; then
        stop_text_encoder_training=$((stop_text_encoder_training * gradient_accumulation_steps))
      fi
      ext_args+=("--stop_text_encoder_training=$stop_text_encoder_training")
    fi
    if [ "$learning_rate_te" ]; then
      ext_args+=("--learning_rate_te=$learning_rate_te")
    fi
  else
    laungh_script="train"
    if [ "$diffusers_xformers" -ne 0 ]; then
      ext_args+=("--diffusers_xformers")
      sdpa=0
    fi
    if [ "$train_text_encoder" -ne 0 ]; then
      cache_text_encoder_outputs=0
      ext_args+=("--train_text_encoder")
      if [ "$learning_rate_te1" -ne 0 ]; then
        ext_args+=("--learning_rate_te1=$learning_rate_te1")
      fi
      if [ "$learning_rate_te2" -ne 0 ]; then
        ext_args+=("--learning_rate_te2=$learning_rate_te2")
      fi
    fi
    if [ "$enable_block_lr" -ne 0 ]; then
      ext_args+=("--block_lr=$block_lr")
    fi
  fi
fi

if [[ "$train_mode" =~ "cn3l" ]]; then
  laungh_script="cn3l"
  if [ "$conditioning_data_dir" ]; then
    ext_args+=("--conditioning_data_dir=$conditioning_data_dir")
  fi
  if [ "$cond_emb_dim" ]; then
    ext_args+=("--cond_emb_dim=$cond_emb_dim")
  fi
fi

if [[ "$train_mode" =~ "sdxl"* ]]; then
  laungh_script="sdxl_$laungh_script"
  if [ "$min_timestep" -ne 0 ]; then
    ext_args+=("--min_timestep=$min_timestep")
  fi
  if [ "$max_timestep" -ne 1000 ]; then
    ext_args+=("--max_timestep=$max_timestep")
  fi
  if [ "$cache_text_encoder_outputs" -ne 0 ]; then
    ext_args+=("--cache_text_encoder_outputs")
    if [ "$cache_text_encoder_outputs_to_disk" -ne 0 ]; then
      ext_args+=("--cache_text_encoder_outputs_to_disk")
    fi
    shuffle_caption=0
    train_unet_only=1
    caption_dropout_rate=0
    caption_tag_dropout_rate=0
  fi
  if [ "$no_half_vae" -ne 0 ]; then
    ext_args+=("--no_half_vae")
    mixed_precision=""
    full_fp16=0
    full_bf16=0
  fi
  if [ "$bucket_reso_steps" -ne 64 ]; then
    ext_args+=("--bucket_reso_steps=$bucket_reso_steps")
  fi
fi

if [ "$dataset_class" ]; then
  ext_args+=("--dataset_class=$dataset_class")
else
  ext_args+=("--train_data_dir=$train_data_dir")
fi

if [ "$vae" ]; then
  ext_args+=("--vae=$vae")
fi

if [ "$is_v2_model" -ne 0 ]; then
  ext_args+=("--v2")
  min_snr_gamma=0
  debiased_estimation_loss=0
  if [ "$v_parameterization" -ne 0 ]; then
    ext_args+=("--v_parameterization")
    ext_args+=("--scale_v_pred_loss_like_noise_pred")
  fi
else
  ext_args+=("--clip_skip=$clip_skip")
fi

if [[ $prior_loss_weight && $prior_loss_weight -ne 1 ]]; then
  ext_args+=("--prior_loss_weight=$prior_loss_weight")
fi

if [ "$network_dim" ]; then
  ext_args+=("--network_dim=$network_dim")
fi

if [ "$network_alpha" ]; then
  ext_args+=("--network_alpha=$network_alpha")
fi

if [ "$training_comment" ]; then
  ext_args+=("--training_comment=$training_comment")
fi

if [ "$persistent_workers" ]; then
  ext_args+=("--persistent_data_loader_workers")
fi

if [ "$max_data_loader_n_workers" ]; then
  ext_args+=("--max_data_loader_n_workers=$max_data_loader_n_workers")
fi

if [ "$shuffle_caption" -ne 0 ]; then
  ext_args+=("--shuffle_caption")
fi

if [ "$weighted_captions" -ne 0 ]; then
  ext_args+=("--weighted_captions")
fi

if [ "$cache_latents" ]; then 
  ext_args+=("--cache_latents")
  if [ "$cache_latents_to_disk" ]; then
    ext_args+=("--cache_latents_to_disk")
  fi
fi

if [ "$output_config" -ne 0 ]; then
  ext_args+=("--output_config")
  ext_args+=("--config_file=$config_file")
fi

if [ "$gradient_checkpointing" -ne 0 ]; then
  ext_args+=("--gradient_checkpointing")
fi

if [ "$save_state" -eq 1 ]; then
  ext_args+=("--save_state")
fi

if [ "$resume" ]; then
  ext_args+=("--resume=$resume")
fi

if [ "$noise_offset" ]; then
  ext_args+=("--noise_offset=$noise_offset")
  if [ "$adaptive_noise_scale" ]; then
    ext_args+=("--adaptive_noise_scale=$adaptive_noise_scale")
  fi
elif [ "$multires_noise_iterations" -ne 0 ]; then
  ext_args+=("--multires_noise_iterations=$multires_noise_iterations")
  ext_args+=("--multires_noise_discount=$multires_noise_discount")
fi

if [ "$network_dropout" -ne 0 ]; then
  enable_lycoris=0
  ext_args+=("--network_dropout=$network_dropout")
  if [ "$scale_weight_norms" -ne 0 ]; then 
    ext_args+=("--scale_weight_norms=$scale_weight_norms")
  fi
  if [ "$enable_dylora" -ne 0 ]; then
    ext_args+=("--network_args")
    if [ "$rank_dropout" -ne 0 ]; then
      ext_args+=("rank_dropout=$rank_dropout")
    fi
    if [ "$module_dropout" -ne 0 ]; then
      ext_args+=("module_dropout=$module_dropout")
    fi
  fi
fi

if [ "$enable_block_weights" -ne 0 ]; then
  ext_args+=("--network_args")
  ext_args+=("down_lr_weight=$down_lr_weight")
  ext_args+=("mid_lr_weight=$mid_lr_weight")
  ext_args+=("up_lr_weight=$up_lr_weight")
  ext_args+=("block_lr_zero_threshold=$block_lr_zero_threshold")

  if [ "$enable_block_dim" -ne 0 ]; then
    ext_args+=("block_dims=$block_dims")
    ext_args+=("block_alphas=$block_alphas")

    if [ "$conv_block_dims" ]; then
      ext_args+=("conv_block_dims=$conv_block_dims")
      
      if [ "$conv_block_alphas" ]; then
        ext_args+=("conv_block_alphas=$conv_block_alphas")
      fi
    elif [ "$conv_dim" -ne 0 ]; then
      ext_args+=("conv_dim=$conv_dim")

      if [ "$conv_alpha" -ne 0 ]; then
        ext_args+=("conv_alpha=$conv_alpha")
      fi
    fi
  fi
elif [ "$enable_lycoris" -ne 0 ]; then
  network_module="lycoris.kohya"
  ext_args+=("--network_args")
  ext_args+=("algo=$algo")

  if [ "$use_scalar" ]; then
    ext_args+=("use_scalar=True")
  fi

  if [ "$train_norm" ]; then
    ext_args+=("train_norm=True")
  fi

  if [ "$algo" != "ia3" ]; then
    if [ "$algo" != "full" ]; then
      if [ "$conv_dim" -ne 0 ]; then
        ext_args+=("conv_dim=$conv_dim")

        if [ "$conv_alpha" -ne 0 ]; then
          ext_args+=("conv_alpha=$conv_alpha")
        fi
      fi

      if [ "$use_tucker" ]; then
        ext_args+=("use_tucker=True")
      fi
    fi

    ext_args+=("preset=$preset")
  fi

  if [ "$dropout" -ne 0 ] && [ "$algo" == "locon" ]; then
    ext_args+=("dropout=$dropout")
  fi

  if [ "$algo" == "lokr" ]; then
    ext_args+=("factor=$factor")
  elif [ "$algo" == "dylora" ]; then
    ext_args+=("block_size=$block_size")
  fi
elif [ "$enable_dylora" -ne 0 ]; then
  network_module="networks.dylora"
  ext_args+=("--network_args")
  ext_args+=("unit=$unit")

  if [ "$conv_dim" -ne 0 ]; then
    ext_args+=("conv_dim=$conv_dim")

    if [ "$conv_alpha" -ne 0 ]; then
      ext_args+=("conv_alpha=$conv_alpha")
    fi
  fi

  if [ "$module_dropout" -ne 0 ]; then
    ext_args+=("module_dropout=$module_dropout")
  fi
elif [ "$enable_lora_fa"  -ne 0 ]; then
  network_module="networks.lora_fa"
elif [ "$enable_oft" -ne 0 ]; then
  network_module="networks.oft"
else
  if [ "$conv_dim" -ne 0 ]; then
    ext_args+=("--network_args")
    ext_args+=("conv_dim=$conv_dim")

    if [ "$conv_alpha" -ne 0 ]; then
      ext_args+=("conv_alpha=$conv_alpha")
    fi
  fi
fi

if [ "$optimizer_type" == "adafactor" ]; then
  ext_args+=("--optimizer_type=$optimizer_type")
  ext_args+=("--optimizer_args")
  ext_args+=("scale_parameter=False")
  ext_args+=("warmup_init=False")
  ext_args+=("relative_step=False")
  lr_warmup_steps=100
fi

if [[ "$optimizer_type" == "DAdapt"* ]]; then
  ext_args+=("--optimizer_type=$optimizer_type")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=0.01")

  if [ "$optimizer_type" == "DAdaptation" ] || [[ "$optimizer_type" == "DAdaptAdam"* ]]; then
    ext_args+=("decouple=True")

    if [ "$optimizer_type" == "DAdaptAdam" ]; then
      ext_args+=("use_bias_correction=True")
    fi
  fi

  lr="1"

  if [ "$unet_lr" ]; then
    unet_lr=$lr
  fi

  if [ "$text_encoder_lr" ]; then
    text_encoder_lr=$lr
  fi
fi

if [[ "$optimizer_type" == "Lion" ]] || [[ "$optimizer_type" == "Lion8bit" ]] || [[ "$optimizer_type" == "PagedLion8bit" ]]; then
  ext_args+=("--optimizer_type=$optimizer_type")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=0.01")
  ext_args+=("betas=.95,.98")
fi

if [ "$optimizer_type" == "AdamW8bit" ]; then
  optimizer_type=""
  ext_args+=("--use_8bit_adam")
fi

if [ "$optimizer_type" == "PagedAdamW8bit" ]; then
  ext_args+=("--optimizer_type=$optimizer_type")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=0.01")
fi

if [ "$optimizer_type" == "Sophia" ]; then
  ext_args+=("--optimizer_type=pytorch_optimizer.SophiaH")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=0.01")
fi

if [ "$optimizer_type" == "Prodigy" ]; then
  ext_args+=("--optimizer_type=$optimizer_type")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=0.01")
  ext_args+=("decouple=True")
  ext_args+=("use_bias_correction=True")
  ext_args+=("d_coef=$d_coef")

  if [ "$lr_warmup_steps" ]; then
    ext_args+=("safeguard_warmup=True")
  fi

  if [ "$d0" ]; then
    ext_args+=("d0=$d0")
  fi

  lr="1"

  if [ "$unet_lr" ]; then
    unet_lr=$lr
  fi

  if [ "$text_encoder_lr" ]; then
    text_encoder_lr=$lr
  fi
fi

if [ "$optimizer_type" == "Adan" ]; then
  ext_args+=("--optimizer_type=pytorch_optimizer.Adan")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=2e-5")
  ext_args+=("max_grad_norm=1.0")
  ext_args+=("adanorm=true")
fi

if [ "$optimizer_type" == "Tiger" ]; then
  ext_args+=("--optimizer_type=pytorch_optimizer.Tiger")
  ext_args+=("--optimizer_args")
  ext_args+=("weight_decay=0.01")
fi

if [ "$unet_lr" ]; then
  if [ "$train_unet_only" -ne 0 ]; then
    train_text_encoder_only=0
    ext_args+=("--network_train_unet_only")
  fi
  ext_args+=("--unet_lr=$unet_lr")
fi

if [ "$text_encoder_lr" ]; then
  if [ "$train_text_encoder_only" -ne 0 ]; then
    ext_args+=("--network_train_text_encoder_only")
  fi
  ext_args+=("--text_encoder_lr=$text_encoder_lr")
fi

if [ "$network_weights" ]; then
  ext_args+=("--network_weights=$network_weights")
fi

if [ "$reg_data_dir" ]; then
  ext_args+=("--reg_data_dir=$reg_data_dir")
fi

if [ "$keep_tokens" ]; then
  ext_args+=("--keep_tokens=$keep_tokens")
fi

if [ "$min_snr_gamma" -ne 0 ]; then
  ext_args+=("--min_snr_gamma=$min_snr_gamma")
elif [ "$debiased_estimation_loss" -ne 0 ]; then
  ext_args+=("--debiased_estimation_loss")
fi

if [ "$ip_noise_gamma" -ne 0 ]; then
  ext_args+=("--ip_noise_gamma=$ip_noise_gamma")
fi

if [ "$wandb_api_key" ]; then
  ext_args+=("--wandb_api_key=$wandb_api_key")
  ext_args+=("--log_with=wandb")
  ext_args+=("--log_tracker_name=$output_name")
fi

if [ "$enable_sample" -ne 0 ]; then
  ext_args+=("--sample_every_n_epochs=$sample_every_n_epochs")
  ext_args+=("--sample_prompts=$sample_prompts")
  ext_args+=("--sample_sampler=$sample_sampler")
fi

if [ "$lr_scheduler_num_cycles" ]; then
  ext_args+=("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
fi

if [ "$base_weights" ]; then
  ext_args+=("--base_weights")

  for base_weight in $base_weights; do
    ext_args+=("$base_weight")
  done

  ext_args+=("--base_weights_multiplier")

  for ratio in $base_weights_multiplier; do
    ext_args+=("$ratio")
  done
fi

if [ "$enable_bucket" -ne 0 ]; then
  ext_args+=("--enable_bucket")
  ext_args+=("--min_bucket_reso=$min_bucket_reso")
  ext_args+=("--max_bucket_reso=$max_bucket_reso")
fi

if [ "$full_fp16" -ne 0 ]; then
  ext_args+=("--full_fp16")
  mixed_precision="fp16"
  save_precision="fp16"
elif [ "$full_bf16" -ne 0 ]; then
  ext_args+=("--full_bf16")
  mixed_precision="bf16"
  save_precision="bf16"
fi

if [ "$mixed_precision" ]; then
  ext_args+=("--mixed_precision=$mixed_precision")
fi

if [ "$network_module" ]; then
  ext_args+=("--network_module=$network_module")
fi

if [ "$gradient_accumulation_steps" -ne 0 ]; then
  ext_args+=("--gradient_accumulation_steps=$gradient_accumulation_steps")
fi

if [ "$lr_warmup_steps" ]; then
  if [ "$gradient_accumulation_steps" -ne 0 ]; then
    lr_warmup_steps=$(($lr_warmup_steps * $gradient_accumulation_steps))
  fi
  ext_args+=("--lr_warmup_steps=$lr_warmup_steps")
fi

if [ "$caption_dropout_every_n_epochs" -ne 0 ]; then
  ext_args+=("--caption_dropout_every_n_epochs=$caption_dropout_every_n_epochs")
fi

if [ "$caption_dropout_rate" -ne 0 ]; then
  ext_args+=("--caption_dropout_rate=$caption_dropout_rate")
fi

if [ "$caption_tag_dropout_rate" -ne 0 ]; then
  ext_args+=("--caption_tag_dropout_rate=$caption_tag_dropout_rate")
fi

if [ "$sdpa" ]; then
  ext_args+=("--sdpa")
else
  ext_args+=("--xformers")
fi


# run train
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/$laungh_script.py" \
  --pretrained_model_name_or_path="$pretrained_model" \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --resolution="$resolution" \
  --max_train_epochs="$max_train_epoches" \
  --learning_rate="$lr" \
  --lr_scheduler="$lr_scheduler" \
  --output_name="$output_name" \
  --train_batch_size="$batch_size" \
  --save_every_n_epochs="$save_every_n_epochs" \
  --save_precision="$save_precision" \
  --seed="$seed" \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as="$save_model_as" \
  --vae_batch_size="$vae_batch_size" "${ext_args[@]}"

echo "Train finished"
read -p "Press Enter to continue" </dev/tty
