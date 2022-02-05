VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(hparams):
    return VOCODERS[hparams['vocoder']]


class BaseVocoder:
    def mel2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2mel(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError
