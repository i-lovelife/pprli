from configs.config import Config

z_dim=128
evaluation_verbose=False

@Config.register('cvae_mi')
class CvaeMiConfig(Config):
    @classmethod
    def make_config(self, NAME):
        config={
            "privater":{
                "type":"cvae_mi",
                "z_dim":z_dim,
                "global_weight":1,
                "rec_x_weight":10,
                "local_weight":1,
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

