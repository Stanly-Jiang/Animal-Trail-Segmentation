# Animal-Trail-Segmentation


## For Model Data Preparation
after successfully running step 1a with correct output - anno and image files in the samples folder, 
you can directly run step1b that creates samples_reformatted

after using step 1b go into samples_reformatted folder and create 2 new folders, "label" and "image"
then place the annotation tif files into the label folder and image tif tiles into the image folder

then run jupyter notebook in miniconda, go to model folder and open the ipynb file, go to the
bottom of the three python files, delete the very bottom box by double pressing "d" and sequentially
run each python script until the third one, where you should see an updating output that adds files to the
samples_reformatted folder and creates the hdf5 file next to the two folders in the samples_reformatted folder
-actually you have to wait sometimes a little bit for the output to come out and update in jupyter