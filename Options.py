class Options:
    def __init__(self):
        self.down_sample_rate = 8000
        self.fft_size = 1024
        self.hop_length = 80
        self.window_length = 1024

        self.num_spec = 513 #from 0hz to 4000hz bin
        self.input_size = 31 #time axis
        self.batch_size = 64
        self.resolution = 16 #1/16 semitone (D2 to B5)
        self.figure_on = False
