import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.mvqar import MVQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "infshots" + sweep_id

n_dims = 2
n_clusters = 2
VOCAB_SIZE = 16_384
SEQ_LEN_MULT = (n_dims + 1) / 2
ADD_GISTS = False
ADD_GISTS_LEN = n_clusters if ADD_GISTS else 0

# 1. First we are going to create the data configuration
train_configs = []
for _n_clusters in [2, 4]:
    for _num_kv_per_c in [4, 8, 16, 32]:
        _ADD_GISTS_LEN = _n_clusters if ADD_GISTS else 0
        num_kv_pairs = _num_kv_per_c * _n_clusters
        input_seq_len = num_kv_pairs * (n_dims + 1) * (2 if _num_kv_per_c >= 256 else 3) + _ADD_GISTS_LEN
        train_configs.append(MVQARConfig(
            vocab_size=VOCAB_SIZE, 
            input_seq_len=input_seq_len, 
            num_examples=20_000, 
            num_kv_pairs=num_kv_pairs, 
            n_dims=n_dims, 
            n_clusters=_n_clusters, 
            add_gists=ADD_GISTS,
        ))

test_configs = []
for _n_clusters in [2, 4, 8]:
    for _num_kv_per_c in [4, 8, 16, 32, 64, 128, 256]:  # n-shots
        _ADD_GISTS_LEN = _n_clusters if ADD_GISTS else 0
        num_kv_pairs = _num_kv_per_c * _n_clusters
        input_seq_len = num_kv_pairs * (n_dims + 1) * (2 if _num_kv_per_c > 128 else 3) + _ADD_GISTS_LEN
        test_configs.append(MVQARConfig(
            vocab_size=VOCAB_SIZE, 
            input_seq_len=input_seq_len, 
            num_examples=20_000, 
            num_kv_pairs=num_kv_pairs, 
            n_dims=n_dims, 
            n_clusters=_n_clusters, 
            add_gists=ADD_GISTS,
        ))

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 64
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir="/scratch/sl12886/tts_icl/zoology/data",
    force_cache=False
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
# models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
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
                group="attn-inf-shots-nogist",
            ),
            slice_keys=[["num_kv_pairs", "n_dims", "n_clusters"]],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/scratch/sl12886/tts_icl/zoology/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)

