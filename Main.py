import os
import argparse
import Options
import FeatureExtraction
class MelodyExtractionJDC():
    def __init__(self):
        self.options = Options.Options()
        self.file_path = None
        self.output_dir = None
        self.gpu_index = None
        self.arg_parser()
    
    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p','--file_path',
                            help='path to input audio (default: %(default)s',
                            type=str , default='test_audio_file.mp4')
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
        
        print("debug")

if __name__ == '__main__':
    melody_extraction = MelodyExtractionJDC()
    melody_extraction.main_controller()
    print("debug")