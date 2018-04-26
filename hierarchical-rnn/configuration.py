from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
from hmlstm import HMLSTMNetwork


class YamlParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)


# Parsing YAML configurations
hparams = YamlParams('config.yml', 'default')

network = HMLSTMNetwork(output_size=hparams.output_size,
                        input_size=hparams.input_size,
                        num_layers=hparams.num_layers,
                        embed_size=hparams.embed_size,
                        out_hidden_size=hparams.out_hidden_size,
                        hidden_state_sizes=hparams.hidden_state_sizes,
                        learning_rate=hparams.learning_rate,
                        task='classification')
