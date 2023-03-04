from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

workspace = "C:/SHARE/projects/WesternHeritage/AgInsurance/data/animal_trails/samples_reformatted/"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGen = trainGenerator(2,workspace,'image','label',data_gen_args,save_to_dir = None) # workspace+'aug')

model = unet()
model_checkpoint = ModelCheckpoint(workspace+'animal_trails.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(myGen,steps_per_epoch=100,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator(workspace+"test/")
results = model.predict(testGene,7,verbose=1)
saveResult(workspace+"test",results)