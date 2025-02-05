from app.local import MaskInfo
from config import get_args
from lib_cream_py import InpaintNN

if __name__ == '__main__':
    args = get_args()
    if args.ui_mode:
        import uvicorn
        from app.server.server import app
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        from local import process
        #todo: use input-original-path
        try:
            model = InpaintNN('models/mosaic.keras' if args.is_mosaic else 'models/bar.keras')
            process(args.input, args.output, MaskInfo(args.mask), model, args.variations, args.is_mosaic, args.clean_up_input_dirs)
        except Exception as e:
            print("[ERROR] {}".format(e))