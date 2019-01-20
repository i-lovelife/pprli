from configs.config import Config

model_name = "vae"
@Config.register(model_name)
class VaeConfig(Config):
    @classmethod
    def make_config(self,
                    z_dim=256,
                    rec_x_weight=300,
                    evaluation_verbose=False):
        config={
            "privater":{
                "type":"vae",
                "z_dim":z_dim,
                "rec_x_weight":rec_x_weight,
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
                "epochs":50,
                "save_model":True
            },
            "evaluaters":[
                {"type":"utility",
                 "z_dim":z_dim,
                 "verbose": evaluation_verbose
                },
                {"type":"private",
                 "z_dim":z_dim,
                 "verbose": evaluation_verbose
                }
            ]
        }
        return Config(config)

