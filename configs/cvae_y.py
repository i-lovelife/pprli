from configs.config import Config

z_dim=64
y_dim=7
p_dim=6
evaluation_verbose=False

@Config.register('cvae_y')
class CvaeYConfig(Config):
    @classmethod
    def make_config(self, NAME):
        config={
            "privater":{
                "type":"cvae_y",
                "z_dim":z_dim,
                "y_dim":y_dim,
                "p_dim":p_dim,
                "rec_x_weight":100,
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

