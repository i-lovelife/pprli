from configs.config import Config

z_dim=32
evaluation_verbose=False

@Config.register('cvae_mi')
class CvaeMiConfig(Config):
    @classmethod
    def make_config(self, NAME):
        config={
            "privater":{
                "type":"cvae_mi",
                "z_dim":z_dim,
                "global_weight":10,
                "local_weight":15,
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
                "type":"keras",
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

