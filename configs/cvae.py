NAME="cvae"
z_dim=64
evaluation_verbose=False
config={
    "privater":{
        "type":"cvae",
        "z_dim":z_dim,
        "rec_x_weight":64*64/10,
        "encrypt_with_noise": True,
        "optimizer":{
            "type": "rmsprop",
            "lr":0.0003,
            "decay":1e-6
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
