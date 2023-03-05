# Animal-Trail-Segmentation

Bash terminal
-Copy path to data:

-data path
"D:\Storage\Random_Stuff\Stanl\DigitizationWork\Animal-Trail-Segmentation\data\data"

Copy path
D:\Storage\Random_Stuff\Stanl\DigitizationWork\Animal-Trail-Segmentation\data\train_data
echo "D:\Storage\Random_Stuff\Stanl\DigitizationWork\Animal-Trail-Segmentation\data\train_data" | sed 's/\\/\//g'
D:/Storage/Random_Stuff/Stanl/DigitizationWork/Animal-Trail-Segmentation/data/train_data

Copy Relative path
data/train_data
data/output_data

Environment creation in venv module in a bash terminal on windows

Create environment named deeplabenv
python -m venv deeplabenv

activate environment & deactivate environment
1. source deeplabenv/Scripts/activate
2. deactivate 

update pip package manager
python.exe -m pip install --upgrade pip

install modules
#cd to requirements text file - cd modelling
#pip install -r requirements.txt

https://pytorch.org/get-started/locally/
Stable, Windows, Pip, Python, CUDA 11.7, retrieve command, very latest version of python may not be compatible:
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

run:
python main_modeling.py

bottom blue banner -> select interpreter python version on your path
venv setup: 
root development directory /d/Storage/Random_Stuff/Stanl/DigitizationWork/Animal-Trail-Segmentation
$ python -m venv venv
$ source venv/Scripts/activate 
-> (venv)
deactivate

Copilot:
Ctr+enter for suggestions, comment, tab/enter

./ means start in the relative current directory 
../ means start in the relative directory but one directory level up 


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