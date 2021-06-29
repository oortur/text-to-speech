# text-to-speech
Text-to-speech system with mel-spectrogram generator and duration predictor

The implementation involves:

- Text encoder model with self-attention and convolution based on paper https://arxiv.org/pdf/1910.10352.pdf
- Duration model to predict phoneme duration in spectrogram frames
- Alignment model for upsampling encoded represantation to spectrogram shape based on Gaussian upsampling from https://arxiv.org/pdf/2010.04301.pdf
- Decoder model to induce mel-spectrograms in similar to encoder manner

The generated mel-spectrograms are then converted to audio format using WaveGlow 
