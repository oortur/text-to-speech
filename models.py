import torch
import torch.nn as nn
import torchtext
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import lj_speech


DATASET_ROOT = Path("./LJ-Speech-aligned/")


class ConvResBlock(nn.Module):
    def __init__(
        self,
        in_features=1024,
        out_features=256,
        dropout=0.15, 
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1dk1 = nn.Conv1d(in_channels=self.in_features, out_channels=self.out_features, kernel_size=1)
        self.conv1dk5 = nn.Conv1d(in_channels=self.in_features, out_channels=self.out_features, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=self.out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x ~ (bs, seq_len, in_features)
        res = self.conv1dk1(x.permute(0, 2, 1))
        x = self.conv1dk5(x.permute(0, 2, 1))
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = x + res
        x = x.permute(0, 2, 1)
        # x ~ (bs, seq_len, out_features)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        n_phonemes,
        embedding_dim=256,
        attn_dim=1024,
        n_attn_heads=4,
        n_blocks=3,
        attn_dropout=0.15,
        res_dropout=0.15,
        padding_index=None,
        # max length of phrase in phonemes = 137
        max_phonemes=150,
    ):
        super().__init__()
        self.n_phonemes = n_phonemes
        self.max_phonemes = max_phonemes
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim

        self.phoneme_embedding = nn.Embedding(num_embeddings=self.n_phonemes, embedding_dim=self.embedding_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=self.max_phonemes, embedding_dim=self.embedding_dim)
        self.padding_index = padding_index

        self.linear_blocks = nn.ModuleList([
            nn.Linear(in_features=self.embedding_dim, out_features=self.attn_dim)
        for _ in range(n_blocks)])

        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.attn_dim, nhead=n_attn_heads, dim_feedforward=self.attn_dim, dropout=attn_dropout)
        for _ in range(n_blocks)])

        self.res_blocks = nn.ModuleList([
            ConvResBlock(in_features=self.attn_dim, out_features=self.embedding_dim, dropout=res_dropout)
        for _ in range(n_blocks)])

    def forward(self, x):
        """ x ~ (bs, seq_len) """
        device = x.device
        bs, seq_len = x.shape[:2]

        # compute padding mask (use self.padding_index)
        mask = None
        if self.padding_index is not None:
            mask = (x == self.padding_index)
        
        # sum up phoneme and positional embeddings
        ph_emb = self.phoneme_embedding(x)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1).to(device)
        pos_emb = self.pos_embedding(pos)
        x = ph_emb + pos_emb

        # attention + conv_res_blocks
        for linear_block, attn_block, res_block in zip(self.linear_blocks, self.attention_blocks, self.res_blocks):
            x = linear_block(x)
            x = x.transpose(0, 1)
            x = attn_block(x, src_key_padding_mask=mask)
            x = x.transpose(0, 1)
            x = res_block(x)

        # x ~ (bs, seq_len, emb_dim)
        return x


class AlignmentModel(nn.Module):
    def __init__(
        self,
        log_sigma=0,
    ):
        super().__init__()
        log_sigma = log_sigma or torch.randn(1).item()
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma), requires_grad=True)
        self.mask_fill_value = -1e10

    def forward(self, emb, durations):
        """
        emb ~ (bs, seq_len, emb_dim)
        durations ~ (bs, seq_len)
        """
        bs, seq_len = emb.shape[:2]
        device = emb.device

        # (bs, seq_len)
        centers = torch.cumsum(durations, dim=1) - durations * 0.5

        # T - max num of ticks per batch
        T = torch.sum(durations, dim=1).int().max().item()

        # (bs, T, 1)
        t = torch.arange(T).repeat(bs, 1).unsqueeze(2).to(device)
        normal = torch.distributions.Normal(loc=centers.unsqueeze(1), scale=torch.exp(self.log_sigma).view(1, 1, 1))
        # (bs, T, seq_len)
        prob = normal.log_prob(t + 0.5)
        mask = (durations == 0).unsqueeze(1)
        prob = prob.masked_fill(mask, self.mask_fill_value)
        w = nn.Softmax(dim=2)(prob)

        # (bs, T, emb_dim)
        x = torch.bmm(w, emb)

        # (bs, T)
        out_mask = t < durations.sum(dim=1).view(bs, 1, 1)
        out_mask.squeeze_(dim=2)
        
        return x, out_mask


class DurationModel(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        rnn_hidden_dim=512,
        n_rnn_layers=3,
        # max_duration is taken empirically, 
        # there are few outliers with larger duration (true max_duration = 254),
        # but let us filter them for stability
        max_duration=80,
        min_duration=1,
        dropout=0.1,
        act='softplus',
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.rnn_hidden_dim, 
            num_layers=n_rnn_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(in_features=2*self.rnn_hidden_dim, out_features=1)
        self.max_duration = nn.Parameter(torch.tensor(max_duration), requires_grad=False)
        self.min_duration = nn.Parameter(torch.tensor(min_duration), requires_grad=False)
        self.act_name = act.lower()
        if self.act_name == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Softplus()
            self.act_name = 'softplus'

    def forward(self, x):
        # x ~ (bs, seq_len, emb_dim)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x.squeeze_(dim=2)
        # x ~ (bs, seq_len)
        if self.act_name == 'sigmoid':
            x = torch.add(torch.multiply(self.act(x), self.max_duration), self.min_duration)
        else:
            x = torch.add(self.act(x), self.min_duration)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        attn_dim=1024,
        spectrogram_dim=80,
        n_attn_heads=4,
        n_blocks=3,
        n_post_processing_steps=5,
        attn_dropout=0.1,
        res_dropout=0.1,
        # max length of phrase in timesteps = 869
        max_timesteps=900,
    ):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim
        self.spectrogram_dim = spectrogram_dim

        self.pos_embedding = nn.Embedding(num_embeddings=self.max_timesteps, embedding_dim=self.embedding_dim)

        self.linear_blocks = nn.ModuleList([
            nn.Linear(in_features=self.embedding_dim, out_features=self.attn_dim)
        for _ in range(n_blocks)])

        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.attn_dim, nhead=n_attn_heads, dim_feedforward=self.attn_dim, dropout=attn_dropout)
        for _ in range(n_blocks)])

        self.res_blocks = nn.ModuleList([
            ConvResBlock(in_features=self.attn_dim, out_features=self.embedding_dim, dropout=res_dropout)
        for _ in range(n_blocks)])

        self.post_processing_blocks = nn.ModuleList([
            ConvResBlock(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=0.)
        for _ in range(n_post_processing_steps)])

        self.intermediate_conv = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.spectrogram_dim, kernel_size=1)
        self.correction_conv = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.spectrogram_dim, kernel_size=1)

    def forward(self, x, mask=None):
        # x ~ (bs, seq_len, emb_dim)
        # mask ~ (bs, seq_len)
        # False elements should be ignored!
        device = x.device
        bs, seq_len = x.shape[:2]
        
        # sum up with positional embeddings
        pos = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1).to(device)
        pos_emb = self.pos_embedding(pos)
        x = x + pos_emb

        # attention + conv_res_blocks
        for linear_block, attn_block, res_block in zip(self.linear_blocks, self.attention_blocks, self.res_blocks):
            x = linear_block(x)
            x = x.transpose(0, 1)
            x = attn_block(x, src_key_padding_mask=~mask)
            x = x.transpose(0, 1)
            x = res_block(x)

        # (bs, seq_len, spectrogram_dim)
        intermediate = self.intermediate_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # post-processing correction
        for post_block in self.post_processing_blocks:
            x = post_block(x)

        # (bs, seq_len, spectrogram_dim)
        corrected = self.correction_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        corrected = corrected + intermediate

        return corrected, intermediate


class TTS(nn.Module):
    def __init__(
        self,
        n_phonemes=56,
        padding_index=1,
    ):
        super().__init__()
        self.encoder = Encoder(n_phonemes=n_phonemes, padding_index=padding_index)
        self.alignment_model = AlignmentModel()
        self.duration_model = DurationModel()
        self.decoder = Decoder()

    def forward(self, phoneme, duration=None):
        """
        phoneme ~ (bs, seq_len)
        duration ~ (bs, seq_len)
        """
        x = self.encoder(phoneme)
        # x ~ (bs, seq_len, emb_dim)

        if duration is None:
            duration = self.duration_model(x)
            # duration ~ (bs, seq_len)

        x, mask = self.alignment_model(x, duration)
        # x ~ (bs, timesteps ,emb_dim)

        corrected, intermediate = self.decoder(x, mask)
        # both ~ (bs, timesteps, spectrogram_dim)

        return corrected, intermediate


def datasets_with_vocab(train_dataset, val_dataset):
    """Prepares torchtext datasets with phoneme vocabulary"""
    DURATION_PAD_TOKEN = 0

    PHONEME = torchtext.legacy.data.Field(
        batch_first=True,
        lower=False,
        # init_token='<bos>', eos_token='<eos>'  # special tokens: beginning/end of sequence
    )
    DURATION = torchtext.legacy.data.Field(
        batch_first=True,
        sequential=True,
        use_vocab=False, 
        pad_token=DURATION_PAD_TOKEN,
    )
    SPECTROGRAM = torchtext.legacy.data.RawField()

    fields = [('phonemes_code', PHONEME), ('phonemes_duration', DURATION), ('spectrogram', SPECTROGRAM)]

    train_examples = [torchtext.legacy.data.Example.fromlist(
        [sample['phonemes_code'], sample['phonemes_duration'], sample['spectrogram']], 
        fields,
    ) for sample in train_dataset]

    val_examples = [torchtext.legacy.data.Example.fromlist(
        [sample['phonemes_code'], sample['phonemes_duration'], sample['spectrogram']], 
        fields,
    ) for sample in val_dataset]

    torchtext_train_dataset = torchtext.legacy.data.Dataset(examples=train_examples, fields=fields)
    torchtext_val_dataset = torchtext.legacy.data.Dataset(examples=val_examples, fields=fields)

    PHONEME.build_vocab(torchtext_train_dataset.phonemes_code)
    return torchtext_train_dataset, torchtext_val_dataset, PHONEME


def dataiterators(
    torchtext_train_dataset, 
    torchtext_val_dataset, 
    device,
    train_batch_size, 
    val_batch_size,
):
    """Prepares data iterators with bucketing"""
    train_iterator = torchtext.legacy.data.BucketIterator(
        dataset=torchtext_train_dataset,
        batch_size=train_batch_size, 
        sort_key=lambda sample: sum(sample.phonemes_duration),
        sort=True,
        device=device,
    )
    val_iterator = torchtext.legacy.data.BucketIterator(
        dataset=torchtext_val_dataset,
        batch_size=val_batch_size, 
        sort_key=lambda sample: sum(sample.phonemes_duration),
        sort=True,
        device=device,
    )
    return train_iterator, val_iterator


class SpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1Loss = nn.L1Loss()
        self.L2Loss = nn.MSELoss()

    def forward(self, spectrogram, corrected, intermediate):
        """
        spectrogram - list of numpy arrays ~ (spec_dim, timesteps)
        corrected, intermediate ~ (bs, timesteps, spec_dim)
        """
        device = corrected.device
        gt = [torch.tensor(spec).transpose(0, 1) for spec in spectrogram]
        gt = nn.utils.rnn.pad_sequence(gt, batch_first=True).to(device)
        pad_mask = (gt != 0)
        corr_loss = self.L1Loss(corrected * pad_mask, gt) + self.L2Loss(corrected * pad_mask, gt)
        inter_loss = self.L1Loss(intermediate * pad_mask, gt) + self.L2Loss(intermediate * pad_mask, gt)
        loss = corr_loss + inter_loss
        return loss


class DurationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, true, pred):
        """true, pred ~ (bs, seq_len)"""
        loss = self.criterion(true.float(), pred)
        return loss


def run_epoch_spectrogram(tts_model, dataloader, optimizer, criterion, phase='train', CLIP_SIZE=1):
    is_train = (phase == 'train')
    if is_train:
        tts_model.train()
    else:
        tts_model.eval()
    
    epoch_loss = 0
    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(dataloader):

            # unpack batch
            phoneme = batch.phonemes_code
            duration = batch.phonemes_duration
            spec = batch.spectrogram

            # make prediction
            corr, inter = tts_model(phoneme, duration)

            # calculate loss
            loss = criterion(spec, corr, inter)
            
            if is_train:
                # make optimization step
                loss.backward()
                nn.utils.clip_grad_norm_(tts_model.parameters(), max_norm=CLIP_SIZE)
                optimizer.step()
                optimizer.zero_grad()

            # log per-batch train metrics
            epoch_loss += loss.item()

            # 
            if i % 100 == 0:
                print(f"Batch #{i}, epoch loss: {epoch_loss}")
            # 

        average_loss = epoch_loss / len(dataloader)
        return average_loss


def run_epoch_duration(tts_model, dataloader, optimizer, criterion, phase='train', CLIP_SIZE=1):
    is_train = (phase == 'train')
    if is_train:
        tts_model.train()
    else:
        tts_model.eval()
    
    epoch_loss = 0
    with torch.set_grad_enabled(is_train):
        for i, batch in enumerate(dataloader):

            # unpack batch
            phoneme = batch.phonemes_code
            true_duration = batch.phonemes_duration

            # make prediction
            emb = tts_model.encoder(phoneme)
            pred_duration = tts_model.duration_model(emb)

            # calculate loss
            loss = criterion(true_duration, pred_duration)
            
            if is_train:
                # make optimization step
                loss.backward()
                nn.utils.clip_grad_norm_(tts_model.duration_model.parameters(), max_norm=CLIP_SIZE)
                optimizer.step()
                optimizer.zero_grad()

            # log per-batch train metrics
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        return average_loss


def train_tts(
    dataset_root: Path = DATASET_ROOT, 
    model_path: str = "./TTS.pth", 
    num_epochs: int = 9,
    num_epochs_duration: int = 10,
    train_batch_size: int = 10, 
    val_batch_size: int = 10,
    lr_spec: float = 3e-5,
    lr_dur: float = 3e-4,
    scheduler_spec_step_size: int = 2,
    scheduler_dur_step_size: int = 5,
    scheduler_factor: int = 0.33,
    device: str = 'cuda',
    return_iterators: bool = False,
):
    """
    Train the TTS system from scratch on LJ-Speech-aligned stored at
    `dataset_root` for `num_epochs` epochs and save the best model to
    (!!! 'best' in terms of audio quality!) "./TTS.pth".

    dataset_root:
        `pathlib.Path`
        The argument for `lj_speech.get_dataset()`.
    """
    train_dataset, val_dataset = lj_speech.get_dataset(dataset_root)

    torchtext_train_dataset, torchtext_val_dataset, PHONEME = datasets_with_vocab(train_dataset, val_dataset)
    N_PHONEMES = len(PHONEME.vocab)
    PHONEME_PAD_IDX = PHONEME.vocab.stoi[PHONEME.pad_token]

    train_iterator, val_iterator = dataiterators(
        torchtext_train_dataset, 
        torchtext_val_dataset, 
        device=device, 
        train_batch_size=train_batch_size, 
        val_batch_size=val_batch_size,
    )
    
    tts_model = TTS(n_phonemes=N_PHONEMES, padding_index=PHONEME_PAD_IDX).to(device)

    # train full TTS model to predict spectrogram
    criterion = SpectrogramLoss()
    optimizer = torch.optim.Adam(tts_model.parameters(), lr=lr_spec)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_spec_step_size, gamma=scheduler_factor)
    best_val_loss = float("+inf")
    writer = SummaryWriter("./logs")

    for epoch in range(num_epochs):
        train_loss = run_epoch_spectrogram(tts_model, train_iterator, optimizer, criterion, phase='train')
        val_loss = run_epoch_spectrogram(tts_model, val_iterator, None, criterion, phase='val')
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(tts_model.state_dict(), model_path)
        
        print(f'TTS Model Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}')
        writer.add_scalars('TTS Model Loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
    writer.close()

    # train Duration model independently
    tts_model = train_duration_model(
        tts_model, 
        train_iterator, 
        val_iterator, 
        num_epochs_duration, 
        model_path, 
        lr_dur,
        scheduler_dur_step_size, 
        scheduler_factor,
    )

    if return_iterators:
        return tts_model, train_iterator, val_iterator
    return tts_model


def train_duration_model(
    tts_model, 
    train_iterator, 
    val_iterator, 
    num_epochs: int = 10, 
    model_path: str = "./TTS.pth",
    lr: float = 3e-4,
    scheduler_step_size: int = 10,
    scheduler_factor: int = 0.33,
):
    criterion = DurationLoss()
    optimizer = torch.optim.Adam(tts_model.duration_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_factor)
    best_val_loss = float("+inf")
    writer = SummaryWriter("./logs")

    # freeze encoder weights
    for params in tts_model.encoder.parameters():
        params.requires_grad = False
    
    for epoch in range(num_epochs):
        train_loss = run_epoch_duration(tts_model, train_iterator, optimizer, criterion, phase='train')
        val_loss = run_epoch_duration(tts_model, val_iterator, None, criterion, phase='val')
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(tts_model.state_dict(), model_path)

        print(f'Duration Model Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}')
        writer.add_scalars('Duration Model Loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
    writer.close()
    
    return tts_model


class TextToSpeechSynthesizer:
    """
    Inference-only interface to the TTS model.

    It is highly recommended to install packages (to avoid troubles with lj_speech text to phonemes conversion):
    !apt-get install espeak
    !apt-get install espeak-ng
    """
    def __init__(self, checkpoint_path: str = "./TTS.pth", dataset_root: Path = DATASET_ROOT):
        """
        Create the TTS model on GPU, loading its weights from `checkpoint_path`.

        checkpoint_path:
            `str`
        """
        print("Initializing TTS model...")
        self.vocoder = lj_speech.Vocoder()

        train_dataset, val_dataset = lj_speech.get_dataset(dataset_root)
        self.PHONEME = datasets_with_vocab(train_dataset, val_dataset)[2]
        N_PHONEMES = len(self.PHONEME.vocab)
        PHONEME_PAD_IDX = self.PHONEME.vocab.stoi[self.PHONEME.pad_token]
        
        self.tts_model = TTS(n_phonemes=N_PHONEMES, padding_index=PHONEME_PAD_IDX).cuda()
        self.tts_model.load_state_dict(torch.load(checkpoint_path))
        self.tts_model.eval()
        print("Ready!")
        
    def synthesize_from_text(self, text):
        """
        Synthesize text into voice.

        text:
            `str`

        return:
        audio:
            `torch.Tensor` or `numpy.ndarray`, shape == (1, t)
        """
        phonemes = lj_speech.text_to_phonemes(text)
        return self.synthesize_from_phonemes(phonemes)

    def synthesize_from_phonemes(self, phonemes, durations=None):
        """
        Synthesize phonemes into voice.

        phonemes:
            `list` of `str`
            ARPAbet phoneme codes.
        durations:
            `list` of `int`, optional
            Duration in spectrogram frames for each phoneme.
            If given, used for alignment in the model (like during
            training); otherwise, durations are predicted by the duration
            model.

        return:
        audio:
            torch.Tensor or numpy.ndarray, shape == (1, t)
        """
        phonemes = [self.PHONEME.vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes = torch.tensor(phonemes).int().unsqueeze(0).cuda()

        # cut into list of tensors of TTS encoder max_phonemes length
        n_phonemes = phonemes.shape[1]
        max_phonemes = self.tts_model.encoder.max_phonemes
        phonemes_list = [phonemes.squeeze(0)[i : i+max_phonemes] for i in range(0, phonemes.shape[1], max_phonemes)]
        
        # pack into batch and process with encoder
        phonemes = nn.utils.rnn.pad_sequence(phonemes_list, batch_first=True, padding_value=self.tts_model.encoder.padding_index)
        encoded = self.tts_model.encoder(phonemes)
        encoded = encoded.reshape(-1, self.tts_model.encoder.embedding_dim)[:n_phonemes, :].unsqueeze(0)

        # calculate durations if they are not provided
        if durations is not None:
            durations = torch.tensor(durations).int().unsqueeze(0).cuda()
        else:
            durations = self.tts_model.duration_model(encoded)
        
        aligned, mask = self.tts_model.alignment_model(encoded, durations)
        
        # cut into list of tensors of TTS decoder max_timesteps length
        n_timesteps = aligned.shape[1]
        max_timesteps = self.tts_model.decoder.max_timesteps
        aligned_list = [aligned.squeeze(0)[i : i+max_timesteps, :] for i in range(0, aligned.shape[1], max_timesteps)]
        mask_list = [mask.squeeze(0)[i : i+max_timesteps] for i in range(0, mask.shape[1], max_timesteps)]
        
        # pack into batch and process with decoder
        aligned = nn.utils.rnn.pad_sequence(aligned_list, batch_first=True)
        mask = nn.utils.rnn.pad_sequence(mask_list, batch_first=True, padding_value=False)
        spectrogram = self.tts_model.decoder(aligned, mask)[0]
        spectrogram = spectrogram.reshape(-1, self.tts_model.decoder.spectrogram_dim)[:n_timesteps, :].transpose(0, 1)

        return self.vocoder(spectrogram)


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'TTS.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'TTS.pth'.
        On Linux (in Colab too), use `$ md5sum TTS.pth`.
        On Windows, use `> CertUtil -hashfile TTS.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'TTS.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    md5_checksum = "62abf7db8cef3497a3abcf86eecf5ee2"    
    google_drive_link = "https://drive.google.com/file/d/1dAdvjhKvagx1qe3GdsVoCGxW6RiMTnPg/view?usp=sharing"

    return md5_checksum, google_drive_link
    