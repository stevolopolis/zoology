import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig
from zoology.data.mvqar import MVQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "inf" + sweep_id

VOCAB_SIZE = 8_192
ADD_GISTS = True

# 1. First we are going to create the data configuration

train_configs = [    
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64 + (4 if ADD_GISTS else 0), num_examples=100_000, num_kv_pairs=4, n_dims=1, n_clusters=4, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128 + (8 if ADD_GISTS else 0), num_examples=20_000, num_kv_pairs=8, n_dims=1, n_clusters=8, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256 + (16 if ADD_GISTS else 0), num_examples=20_000, num_kv_pairs=16, n_dims=1, n_clusters=16, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256 + (32 if ADD_GISTS else 0), num_examples=20_000, num_kv_pairs=32, n_dims=1, n_clusters=32, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256 + (64 if ADD_GISTS else 0), num_examples=20_000, num_kv_pairs=64, n_dims=1, n_clusters=64, add_gists=ADD_GISTS),
]
test_configs = [
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64 + (4 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=4, n_dims=1, n_clusters=4, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64 + (8 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=8, n_dims=1, n_clusters=8, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64 + (16 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=16, n_dims=1, n_clusters=16, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128 + (32 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=32, n_dims=1, n_clusters=32, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256 + (64 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=64, n_dims=1, n_clusters=64, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512 + (128 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=128, n_dims=1, n_clusters=128, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024 + (256 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=256, n_dims=1, n_clusters=256, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=2048 + (512 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=512, n_dims=1, n_clusters=512, add_gists=ADD_GISTS),
    MVQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=4096 + (1024 if ADD_GISTS else 0), num_examples=1_000, num_kv_pairs=1024, n_dims=1, n_clusters=1024, add_gists=ADD_GISTS),
]

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir="/scratch/sl12886/tts_icl/zoology/data",
    force_cache=True
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)


from zoology.experiments.models_repo import add_attention, add_based, add_mamba2, add_rwkv7, add_delta_net, add_gla, add_gated_delta_net, add_deepseek_nsa, add_gistsa, add_gsa

models = add_based(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs)
# models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gistsa(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gsa(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs)

# convenience for filtering out 
included = ["attention"]
models = [m for m in models if any([i in m.name for i in included])]


# 3. Finally we'll create a train config for each
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-{model.d_model}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="zoology",
                entity="stevenluots",
                group="attn-inf-pairs-withgist",
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/scratch/sl12886/tts_icl/zoology/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)

