from local import MaskInfo
from config import get_args
from lib_cream_py import InpaintNN

if __name__ == '__main__':
    args = get_args()
    if args.ui_mode:
        import uvicorn
        import server.server as server
        server.check_disconnect = args.require_keep_connection
        uvicorn.run(server.app, host=args.host, port=args.port)
    else:
        from local import process, CliLogger
        #todo: use input-original-path
        try:
            logger = CliLogger()
            model = InpaintNN('./models/mosaic.keras' if args.is_mosaic else './models/bar.keras', logger=logger)
            process(args.input, args.output, MaskInfo(args.mask), model, args.variations, args.is_mosaic, args.clean_up_input_dirs, logger)
        except Exception as e:
            print("[ERROR] {}".format(e))