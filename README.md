# colorization
[Convolutional Neural Network based Image Colorization using OpenCV](https://www.learnopencv.com/convolutional-neural-network-based-image-colorization-using-opencv/)

## Input
Input can be any grayscale image or video. This project has greyscaleImage.png and Mona_Lisa_GS2.jpeg as test images, and greyscaleVideo.mp4 as test video. 

## Execution (Run in terminal)

1. Jump to the project file location

2. Run the getModels.sh file from command line to download the needed model files
  ```
	sudo chmod a+x getModels.sh
	./getModels.sh
  ```
3. This project includes two models of colorization
  * colorization_release_v2_norebal.caffemodel

    which has color rebalancing that contributes towards getting more vibrant and saturated colors in the output 
  * colorization_release_v2.caffemodel 
  
  
  In this part of the colorizeImage.py file, please comment out one line of model quote and try out the other model.  
  ```
  # Specify the paths for the 2 model files
  protoFile = "./models/colorization_deploy_v2.prototxt"
  # Model with color rebalancing that contributes towards getting more vibrant and saturated colors in the output
  weightsFile = "./models/colorization_release_v2.caffemodel"
  # ⬇Model without color relalancing
  # weightsFile = "./models/colorization_release_v2_norebal.caffemodel"
  ```
  
  To differenciate the outputs of diffrent models, please comment out one and try out the other. 
  ```
  outputFile = args.input[:-4]+'_colorized.png'  # save
  # outputFile = args.input[:-4]+'_norebal_colorized.png'  # save
  cv.imwrite(outputFile, (img_bgr_out*255).astype(np.uint8))
  ```

4. Commandline usage to colorize

  a single image:
  
    
    python3 colorizeImage.py --input greyscaleImage.png
    
    
  a video file:
    
    
    python3 colorizeVideo.py --input greyscaleVideo.mp4
    
## Output
![Output example](https://www.learnopencv.com/wp-content/uploads/2018/07/color-rebalanced-colorization-example-house-scenery.png)

If you run both model using greyscaleImage.png for example, you can find three images above that are seperately named as greyscaleImage.png, greyscaleImage_colorized.png and greyscaleImage_norebal_colorized.png. 
