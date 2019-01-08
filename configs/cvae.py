from configs.config import Config

z_dim=64
evaluation_verbose=False

@Config.register('cvae')
class CvaeConfig(Config):
    @classmethod
    def make_config(cls):
        config={
            "privater":{
                "type":"cvae",
                "z_dim":z_dim,
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
                }
            ]
        }
        return cls(config)
    
    @classmethod
    def tune_config(cls):
        configs = []
        for z_dim in [32]:#[32, 64, 128, 256, 512]:
            for rec_x_weight in [10]:#[1, 10, 30, 100, 300, 500, 1000, 3000, 9000, 15000]:
                name = f'{cls.__name__}-z_dim{z_dim}-rec_x{rec_x_weight}'
                config = cls.make_config()
                config.config['privater']['z_dim'] = z_dim
                config.config['privater']['rec_x_weight'] = rec_x_weight
                configs.append((name, config))
        return configs

