# Usage
## I. Decensoring bar censors

For each image you want to decensor, using image editing software like Photoshop or GIMP to color the areas you want to decensor the green color (0,255,0), which is a very bright green color and be sure that the picture you use has RGB color Mode not Indexed Color Mode.

*I strongly recommend you use the pencil tool and NOT the brush tool.*

*If you aren't using the pencil tool, BE SURE TO TURN OFF ANTI-ALIASING on the tool you are using.*

I personally use the wand selection tool with anti-aliasing turned off to select the censored regions. I then expand the selections slightly to completely cover the censored areas, pick the color (0,255,0), and use the paint bucket tool on the selected regions.

To expand selections in Photoshop, do Selection > Modify > Expand or Contract.

To expand selections in GIMP, do Select > Grow.

Save these images in the PNG format to the "decensor_input" folder. **They MUST be in PNG format.**

### A. Using the binary

Start a web server and open http://127.0.0.1:8000 in the browser
```
$ binary-name --ui-mode 
```
```
$ binary-name --ui-mode --host 127.0.0.1 --port 8000
```
Or use the cli
```
$ binary-name --decensor_input_path='path/to/input'
```

```
usage: main.py [-h] [--decensor_input_path DECENSOR_INPUT_PATH] [--decensor_input_original_path DECENSOR_INPUT_ORIGINAL_PATH]
               [--decensor_output_path DECENSOR_OUTPUT_PATH] [--clean-up-input-dirs] [--mask MASK] [--is_mosaic IS_MOSAIC] [--variations {1,2,4}] [--ui-mode]
               [--port PORT] [--host HOST]

options:
  -h, --help            show this help message and exit
  --decensor_input_path DECENSOR_INPUT_PATH
                        input images with censored regions colored green to be decensored by decensor.py path
  --decensor_input_original_path DECENSOR_INPUT_ORIGINAL_PATH
                        input images with no modifications to be decensored by decensor.py path
  --decensor_output_path DECENSOR_OUTPUT_PATH
                        output images generated from running decensor.py path
  --clean-up-input-dirs
                        whether to delete all image files in the input directories when decensoring is done
  --mask MASK           red channel of mask color in decensoring
  --is_mosaic IS_MOSAIC
                        true if image has mosaic censoring, false otherwise
  --variations {1,2,4}  number of decensor variations to be generated
  --port [DEFAULT: 8000]
  --host [DEFAULT: 127.0.0.1]
```

### B. Running from scratch
```
$ python app/main.py -h
```

## II. Decensoring mosaic censors

As with decensoring bar censors, perform the same steps of coloring the censored regions green and putting the colored image into the "decensor_input" folder.

In addition, move the original, uncolored images into the "decensor_input_original" folder. Ensure each original image has the same names as their corresponding colored version in the "decensor_input" folder.

For example, if the original image is called "mermaid.jpg," then you want to put this image in the "decensor_input_original" folder and, after you colored the censored regions, name the colored image "mermaid.png" and move it to the "decensor_input" folder.

### A. Using the binary

Decensor the images by double-clicking on the decensor_mosaic file.

### B. Running from scratch

TODO: write documentation

Decensored images will be saved to the "decensor_output" folder. Decensoring takes a few minutes per image.

## III. Decensoring with the user interface

To be implemented.

# Troubleshooting

If you have difficulties getting DeepCreamPy to decensor, go [here](TROUBLESHOOTING.md).
