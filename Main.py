import os
import argparse
import Options
import FeatureExtraction
import JDCModel
from scipy.signal import medfilt
import numpy as np
class MelodyExtractionJDC():
    def __init__(self):
        self.options = Options.Options()
        self.file_path = None
        self.output_dir = None
        self.gpu_index = None
        self.pitch_range = np.arange(38, 83 + 1.0/self.options.resolution, 1.0/self.options.resolution)
        self.pitch_range = np.concatenate([np.zeros(1), self.pitch_range])
        self.min_pitch_midi = 38
        self.max_pitch_midi = 83
        self.arg_parser()
        self.path_output = self.output_dir + '/pitch_'+self.file_path.split('/')[-1]+'.txt'
    
    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p','--file_path',
                            help='path to input audio (default: %(default)s',
                            type=str , default='kim_test_1.mp3')
        parser.add_argument('-gpu', '--gpu_index',
                            help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s',
                            type=int, default=None)
        parser.add_argument('-o', '--output_dir',
                            help='Path to output folder (default: %(default)s',
                            type=str, default='./results/')
        self.file_path = parser.parse_args().file_path
        self.output_dir = parser.parse_args().output_dir
        self.gpu_index = parser.parse_args().gpu_index

    def main_controller(self):
        if self.gpu_index is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_index)

        feature_extraction = FeatureExtraction.FeatureExtraction(self.file_path)
        x_test , x_spectro = feature_extraction.spectro_extraction()
        
        model = JDCModel.JDCModel(self.options).get_jdc_model()
        model.load_weights('./weights/ResNet_joint_add_L(CE_G).hdf5')
        
        y_predict = model.predict(x_test, batch_size=self.options.batch_size, verbose=1)
        
        batch_size = y_predict[0].shape[0]
        num_frame_of_batch = y_predict[0].shape[1]
        total_num_of_pitch = batch_size * num_frame_of_batch
        y_predict = np.reshape(y_predict[0],(total_num_of_pitch,y_predict[0].shape[2]))
        output_pitch = np.zeros(total_num_of_pitch)

        for i in range (total_num_of_pitch):
            index_predict = np.argmax(y_predict[i])
            pitch_midi = self.pitch_range[np.int32(index_predict)]
            if pitch_midi >= self.min_pitch_midi and pitch_midi <= self.max_pitch_midi:
                output_pitch[i] = 2 ** ((pitch_midi - 69) / 12.) * 440

        output_pitch = medfilt(output_pitch, 5)

        if not os.path.exists(os.path.dirname(self.path_output)):
            os.makedirs(os.path.dirname(self.path_output))

        file_to_write = open(self.path_output, 'w')
        for j in range(len(output_pitch)):
            output_pitch_element = "%.2f %.4f\n" % (0.01 * j, output_pitch[j])
            file_to_write.write(output_pitch_element)
        file_to_write.close()

if __name__ == '__main__':
    melody_extraction = MelodyExtractionJDC()
    melody_extraction.main_controller()
    print("debug")