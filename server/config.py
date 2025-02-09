import argparse


def get_args():
    parser = argparse.ArgumentParser(description='')

    # Input output folders settings
    parser.add_argument('-i', '--input', default='./decensor_input/',
                        help='input images with censored regions will be decensored')
    parser.add_argument('--input-original-path', default='./decensor_input_original/',
                        help='input images with no modifications to be decensored by decensor.py path')
    parser.add_argument('-o', '--output', default='./decensor_output/', help='output images folder')
    parser.add_argument('-c', '--clean-up-input-dirs', dest='clean_up_input_dirs', action='store_true', default=False,
                        help='whether to delete all image files in the input directories when decensoring is done')

    # Decensor settings
    parser.add_argument('-m', '--mask', default="rgb-0,255,0",
                        help='mask string file-{suffix} or rgb-{r,g,b} example: rgb-0,255,0 or file-_mask')  # todo: or detect
    parser.add_argument('--is-mosaic', action='store_true', help='true if image has mosaic censoring, false otherwise')
    parser.add_argument('--variations', type=int, choices=[1, 2, 4], default=1,
                        help='number of decensor variations to be generated')

    # Other settings
    parser.add_argument('--ui-mode', action='store_true',
                        help='if you want ui mode, if missing: command line interface')
    parser.add_argument('-p', '--port', type=int, default=8000, help='Server port')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Server host')
    parser.add_argument('--require-keep-connection', action='store_true', help='If this flag is set it requires the user to keep the connection. Decensor requests will be removed if the client disconnects.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    get_args()
