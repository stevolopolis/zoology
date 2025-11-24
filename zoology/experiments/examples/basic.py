from logging import Logger
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig



config = TrainConfig(
    logger=LoggerConfig(
        project_name="zoology",
        entity="stevenluots"
    ),
    data=DataConfig(
        # cache_dir="/path/to/cache/dir"  TODO: add this
        train_configs=[
            MQARConfig(
                num_examples=10_000,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=1_000,
                vocab_size=256,
                input_seq_len=64,
                num_kv_pairs=4
            )
        ]
    ),
    model=ModelConfig(
        vocab_size=256,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.based.Based",
            kwargs={
                "l_max": 64,
                "feature_dim": 8,
                "feature_name": "taylor_exp",
                "num_key_value_heads": 1,
                "num_heads": 1,
                "train_view": "quadratic",
            }
        ),
        block_type="TransformerBlock"
    ),
    
)

configs = [config]