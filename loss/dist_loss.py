def normloss(p):
    def norm_loss(weight=0.1):
        def loss_fn(parameters):
            return parameters.abs().pow(p).sum().pow(1/p) * weight
        return loss_fn
    return norm_loss

L2loss = normloss(2)
L1loss = normloss(1)
