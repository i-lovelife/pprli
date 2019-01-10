from configs.config import Config

model_name = "ad_cvae"
@Config.register('ad_cvae')
class AdCvaeConfig(Config):
    @classmethod
    def make_config(cls,
                    z_dim=256,
                    rec_x_weight=100,
                    evaluation_verbose=False):
        config={
            "privater":{
                "type": model_name,
                "z_dim":z_dim,
                "rec_x_weight":rec_x_weight,
                "prior_weight":1,
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
                "d_iter":2,
                "epochs":30
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
        return cls(config)
    
    @classmethod
    def tune_config(cls):
        configs = []
        for z_dim in [32, 64, 128, 256]:
            for rec_x_weight in [100, 300, 500, 1000, 3000, 15000]:
                name = f'{cls.__name__}-z_dim{z_dim}-rec_x{rec_x_weight}'
                config = cls.make_config(z_dim=z_dim, rec_x_weight=rec_x_weight)
                configs.append((name, config))
        return configs

