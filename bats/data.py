from pathlib import Path

import fairseq as fairseq
import fairseq.utils
import numpy as np
import soundfile
import torch
from fairseq.tasks import audio_finetuning
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, SequenceFeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from scipy import signal

CLASSES = ['Pipkuhlii',
           'Tadaridaten',
           # 'Taphnudiventr',
           'Rhinopomamicrophyllum',
           'Pippip',
           'Rhinopomacystops',
           'Taphnudiventris',
           'Tnudiventr']


# CLASSES = [item.upper() for item in CLASSES]


class BatsDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        path = cfg.dataset.path
        self.species = [item.upper() for item in CLASSES]
        self.data = self.load_dataset(path=path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(cfg.model.encoder_checkpoint)

    def load_dataset(self, path):
        out = []
        files = Path(path).glob('**/*.WAV')
        for file in files:
            if file.parent.stem.upper() in self.species:
                y, sr = soundfile.read(file.as_posix())
                Wn = 10000 / (sr / 2)
                b, a = signal.butter(4, Wn, btype='high', output='ba', fs=sr)
                filtered = signal.lfilter(b=b, a=a, x=y)

                out.append(dict(audio=filtered, sampling_rate=16000, species=file.parent.stem.upper(), file=file))
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = self.feature_extractor(x['audio'], sampling_rate=x['sampling_rate'], padding=True, return_tensors='pt',
                                   species=x['species'], target=list(self.species).index(self.data[index]['species']))[
            'input_values']
        return {'input_values': x[0], 'target': list(self.species).index(self.data[index]['species'])}


class DataCollatorForWav2Vec2Pretraining:  # copied from transformers/examples/pytorch/speech-pretraining
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices for self-supervised
    pretraining. Args: model (:class:`~transformers.Wav2Vec2ForPreTraining`): The Wav2Vec2 model used for
    pretraining. The data collator needs to have access to config and ``_get_feat_extract_output_lengths`` function
    for correct padding. feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`): The processor used for
    processing the data. padding (:obj:`bool`, :obj:`str` or
    :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`): Select a
    strategy to pad the returned sequences (according to the model's padding side and padding index) among: *
    :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided). * :obj:`'max_length'`: Pad to a maximum length specified with the argument
    :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided. *
    :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths). max_length (:obj:`int`, `optional`): Maximum length of the ``input_values`` of the returned list and
    optionally padding length (see above). pad_to_multiple_of (:obj:`int`, `optional`): If set will pad the sequence
    to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA
    hardware with compute capability >= 7.5 (Volta).
    """

    def __init__(self, model, feature_extractor, padding, max_length=None, pad_to_multiple_of=None, config=None):
        self.config = config
        self.model = model
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(
            torch.tensor(batch["input_values"].shape[-1]))
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.config.mask_time_prob,
            self.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )
        mask_time_indices = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        src = batch['input_values']

        return src, mask_time_indices, batch.data['target']


class SpectrogramDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        path = cfg.dataset.path
        self.species = [item.upper() for item in CLASSES]
        self.data = self.load_dataset(path=path)

    def load_dataset(self, path: str):

        out = []
        files = Path(path).glob('**/*.WAV')
        for file in files:
            if file.parent.stem.upper() in self.species:
                y, sr = soundfile.read(file.as_posix())
                y = signal.resample(y, 20000)
                Wn = 10000 / (sr / 2)
                b, a = signal.butter(4, Wn, btype='high', output='ba', fs=sr)
                filtered = signal.lfilter(b=b, a=a, x=y)
                f, t, Sxx = signal.spectrogram(filtered, sr)

                out.append(
                    dict(spect=torch.tensor(Sxx).float()[np.newaxis, ...], species=file.parent.stem.capitalize(),
                         file=file))
        return out

    def __getitem__(self, item):
        return dict(spect=self.data[item]['spect'], target=CLASSES.index(self.data[item]['species']))

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from transformers import Wav2Vec2Model, Wav2Vec2Config

    cfg = OmegaConf.load('configs/wav2vec2-pretraining.yaml')
    # Wav2Vec2Model.from_pretrained('/mnt/drive1/home/ohaddoron1/Projects/bats/models/audio_base_ls_960h.pt')
    cp_path = '/mnt/drive1/home/ohaddoron1/Projects/bats/models/wav2vec_small_960h.pt'
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ['/mnt/drive1/home/ohaddoron1/Projects/bats/models/wav2vec_small.pt.1'])
    model = model[0]

    feature_extractor = Wav2Vec2FeatureExtractor()
    w2v_config = Wav2Vec2Config(mask_time_length=2)
    dataset = BatsDataset(cfg)
    collate_fn = DataCollatorForWav2Vec2Pretraining(model, feature_extractor, padding='longest', config=w2v_config)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    itr = iter(loader)
    sample = next(itr)
    print(sample)
    model.forward(sample[0], sample[1])
