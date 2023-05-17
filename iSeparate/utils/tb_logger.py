from torch.utils.tensorboard import SummaryWriter


class TBLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TBLogger, self).__init__(logdir)

    def log_training(self, loss, learning_rate, iteration):
        for k, v in loss.items():
            if k == "iteration" or k == "best_model" or k == "it":
                continue
            self.add_scalar(f"training/{k}", v, iteration)
        # self.add_scalar("grad.norm", grad_norm, iteration)
        if learning_rate is not None:
            self.add_scalar("learning/rate", learning_rate, iteration)
        self.flush()

    def log_audios(self, audios, iteration):
        self.add_audio("mixture_wav", audios["mixture"], iteration)
        for i in range(audios["target"].shape[0]):
            self.add_audio(
                "target/{}".format(audios["target_names"][i]),
                audios["target"][i],
                iteration,
            )
            self.add_audio(
                "predicted/{}".format(audios["target_names"][i]),
                audios["prediction"][i],
                iteration,
            )
