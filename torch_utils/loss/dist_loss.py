def normloss(p):
    def norm_loss(weight):
        def loss_fn(parameters):
            return parameters.abs().pow(p).sum().pow(1/p)

L2loss = normloss(2)
L1loss = normloss(1)
