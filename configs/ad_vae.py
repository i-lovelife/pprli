from configs.config import Config

z_dim=64
evaluation_verbose=False

@Config.register('ad_vae')
class AdVaeConfig(Config):
    def __init__(self, config):
        self.config  = config
    @classmethod
    def make_config(self, NAME):
        config={
            "privater":{
                "type":"ad_vae",
                "z_dim":z_dim,
                "rec_x_weight":64*64/10,
                "prior_weight": 1,
                "encrypt_with_noise": True,
                "optimizer":{
                    "type": "adam",
                    "lr":0.0003,
                }
            },
            "dataset":{
                "type":"ferg"
            },
            "trainer":{
                "type":"adv",
                "epochs":100
            },
            "evaluaters":[
                {"type":"utility",
                 "z_dim":z_dim,
                 "verbose": evaluation_verbose
                },
                {"type":"private",
                 "z_dim":z_dim,
                 "verbose": evaluation_verbose
                },
                {"type":"reconstruction",
                 "base_dir": NAME
                }
            ]
        }
        return Config(config)
