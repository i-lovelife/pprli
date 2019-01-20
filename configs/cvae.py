from configs.config import Config

model_name = "cvae"
@Config.register(model_name)
class CvaeConfig(Config):
    @classmethod
    def make_config(cls,
                    z_dim=256,
                    rec_x_weight=100,
                    evaluation_verbose=False,
                    evaluation_epoch=10):
        config={
            "privater":{
                "type":model_name,
                "z_dim":z_dim,
                "random_label":False,
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
                 "epochs":evaluation_epoch,
                 "verbose": evaluation_verbose
                },
                {"type":"private",
                 "epochs":evaluation_epoch,
                 "verbose": evaluation_verbose
                },
                {"type":"ssim",
                 "epochs": evaluation_epoch,
                 "verbose": evaluation_verbose
                },
                {"type":"ndm",
                 "epochs": evaluation_epoch,
                 "verbose": evaluation_verbose
                }
            ]
        }
        return cls(config)
    
    @classmethod
    def tune_config(cls):
        configs = []
        for z_dim in [32, 64, 128, 256, 512]:
            for rec_x_weight in [1, 10, 30, 100, 300, 500, 1000, 3000, 9000, 15000]:
                name = f'{cls.__name__}-z_dim{z_dim}-rec_x{rec_x_weight}'
                config = cls.make_config(z_dim=z_dim, rec_x_weight=rec_x_weight)
                configs.append((name, config))
        return configs

