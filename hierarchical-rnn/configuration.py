from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
import argparse
from hmlstm import HMLSTMNetwork, prepare_inputs


class YamlParams(HParams):
    def __init__(self, yaml_fn, config_name):
        super().__init__()
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)
                print("{}: {}".format(k, v))

    def pre_inputs(self, text_path, train=True):
        if not text_path:
            raise Exception("define text_path")
        if self.quick_dev:
            self.num_batches = 3
            self.valid_after_step = 1
            self.output_dir = './quickdev_output/'
        if train:
            return prepare_inputs(batch_size=self.batch_size,
                                  num_batches=self.num_batches,
                                  truncate_len=self.truncate_len,
                                  step_size=self.step_size,
                                  text_path=text_path)
        else:
            num_batches = 500
            if self.quick_dev:
                num_batches = 3
            return prepare_inputs(batch_size=1,
                                  num_batches=num_batches,
                                  truncate_len=self.truncate_len,
                                  step_size=self.truncate_len,
                                  text_path=text_path)

    def gen_network(self):
        return HMLSTMNetwork(output_size=self.output_size,
                             input_size=self.input_size,
                             num_layers=self.num_layers,
                             embed_size=self.embed_size,
                             out_hidden_size=self.out_hidden_size,
                             hidden_state_sizes=self.hidden_state_sizes,
                             learning_rate=self.learning_rate,
                             task='classification',
                             output_dir=self.output_dir)


def select_config():
    # Parsing YAML configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', dest='config', default='default',
                        help="select configuration type in config.yml")
    options = parser.parse_args()
    return YamlParams('config.yml', options.config)
