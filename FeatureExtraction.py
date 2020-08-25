from madmom.audio.signal import *
import librosa
import Options
class FeatureExtraction():
    def __init__(self,file_path):
        self.options = Options.Options()
        self.file_path = file_path
        self.input_win_size = self.options.input_size
        self.frames_data_mean = np.load('./x_data_mean_total_31.npy')
        self.frames_data_std = np.load('./x_data_std_total_31.npy')
        self.alpha = 0.0001
        
    def spectro_extraction(self):
        downsampled_signal = Signal(self.file_path, sample_rate=self.options.down_sample_rate,dtype=np.float32,num_channels=1)
        stft_output = librosa.core.stft(downsampled_signal, n_fft = self.options.fft_size, 
                                        hop_length = self.options.hop_length, win_length = self.options.window_length)
        spectro = np.abs(stft_output)
        spectro = librosa.core.power_to_db(spectro,ref=np.max)
        num_time_frames = spectro.shape[1]

        padd_num = self.input_win_size - (num_time_frames % self.input_win_size)

        if padd_num != self.input_win_size:
            padding_feature = np.zeros(shape=(spectro.shape[0],padd_num))
            spectro = np.concatenate((spectro,padding_feature),axis=1)
            num_time_frames = num_time_frames + padd_num
        
        split_by_input_window_size = []

        for i in range(0,num_time_frames,self.input_win_size):
            split = spectro[:,i:i+self.input_win_size].T
            split_by_input_window_size.append(split)
        split_by_input_window_size = np.array(split_by_input_window_size)

        split_by_input_window_size = (split_by_input_window_size - self.frames_data_mean)/(self.frames_data_std+self.alpha)
        split_by_input_window_size = split_by_input_window_size[:, :, :, np.newaxis]
        
        return split_by_input_window_size,spectro

