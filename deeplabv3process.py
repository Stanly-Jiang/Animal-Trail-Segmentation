import os
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from shapely.geometry import LineString
from skimage.morphology import skeletonize
from skimage import measure
from process import Process
from prediction import Prediction
from skeletonize import Skeletonize

#set up file paths
input_file = "./input/input.tif" #input raster file, change name to match your input file
output_file = "./output/output.shp" #output shapefile
model_file = "./models/model.pt" #model file, change name to match your model file

#set up tile size
tile_size = 512

#set up threshold
threshold = 0.5

#set up input directory for prediction tiles
input_dir = "tiles"

#load model
model = torch.load(model_file)

#initialize process
process = Process()
prediction = Prediction(model)
skeletonize = Skeletonize(threshold)

#open input file and get metadata
with rasterio.open(input_file) as src:
    profile = src.profile
    height = src.height
    width = src.width
    count = src.count

#calculate number of tiles needed
num_tiles_x = int(np.ceil(width / tile_size))
num_tiles_y = int(np.ceil(height / tile_size))

#loop through tiles
for i in range(num_tiles_x):
    for j in range(num_tiles_y):
        #calculate tile bounds
        x_min = i * tile_size
        x_max = min((i + 1) * tile_size, width)
        y_min = j * tile_size
        y_max = min((j + 1) * tile_size, height)
        bounds = (x_min, y_min, x_max, y_max)

        #read tile from input file
        with rasterio.open(input_file) as src:
            tile = src.read(window=Window.from_slices((y_min, y_max), (x_min, x_max)))

        #run tile through prediction model
        tile = prediction.predict(tile)

        #run tile through skeletonization
        tile = skeletonize.skeletonize(tile)

        #save tile to output directory
        with rasterio.open(os.path.join(input_dir, f"tile_{i}_{j}.tif"), "w", **profile) as dst:
            dst.write(tile, 1)
    
#convert skeletonized tiles to vector
process.process(input_dir, output_file)

#save vector output as polyline shapefile
process.save(output_file)

#delete skeletonized tiles
for file in os.listdir(input_dir):
    os.remove(os.path.join(input_dir, file))

#Process class from process.py file that converts skeletonized tiles to vector output and saves as polyline shapefile
import os
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform
from functools import partial
import pyproj
import fiona
from fiona.crs import from_epsg

class Process:
    def __init__(self):
        pass

    def process(self, input_dir, output_file):
        #loop through skeletonized tiles
        for file in os.listdir(input_dir):
            #read tile
            with rasterio.open(os.path.join(input_dir, file)) as src:
                tile = src.read(1)

            #convert tile to vector
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                    shapes(
                        tile,
                        transform=src.transform
                    )
                )
            )

            #save vector output
            with fiona.open(output_file, "a", crs=from_epsg(3857), driver="ESRI Shapefile", schema={'geometry': 'LineString', 'properties': {'raster_val': 'int'}}) as dst:
                for result in results:
                    dst.write(result)

    def save(self, output_file):
        #open vector output
        with fiona.open(output_file, "r") as src:
            #get coordinate reference system
            crs = src.crs

            #get schema
            schema = src.schema

            #get features
            features = [feature for feature in src]

        #convert features to polylines
        features = [self.to_polyline(feature) for feature in features]

        #save vector output as polyline shapefile
        with fiona.open(output_file, "w", crs=crs, driver="ESRI Shapefile", schema=schema) as dst:
            for feature in features:
                dst.write(feature)

    def to_polyline(self, feature):
        #get geometry
        geometry = feature["geometry"]

        #convert geometry to shapely shape
        shape = self.to_shape(geometry)

        #convert shapely shape to polyline
        polyline = self.to_linestring(shape)

        #convert polyline to geometry
        geometry = self.to_geometry(polyline)

        #update feature
        feature["geometry"] = geometry

        return feature

    def to_shape(self, geometry):
        #convert geometry to shapely shape
        shape = mapping(geometry)

        return shape

    def to_linestring(self, shape):
        #convert shapely shape to polyline, projected coordinate system is WGS 1984 Web Mercator (auxiliary sphere)
        polyline = transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init="epsg:3857"),
                pyproj.Proj(init="epsg:3857", preserve_units=True)
            ),
            shape
        )

#Prediction class from prediction.py file that runs tile through prediction model, utilizing gpu if available
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class Prediction:
    def __init__(self, model_file):
        #set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #load model
        self.model = torch.load(model_file, map_location=self.device)
        self.model.eval()

    def predict(self, tile):
        #convert tile to tensor
        tensor = torch.from_numpy(tile).float()

        #add batch dimension
        tensor = tensor.unsqueeze(0)

        #move tensor to gpu if available
        tensor = tensor.to(self.device)

        #run tensor through model
        output = self.model(tensor)

        #convert output to numpy array
        output = output.detach().cpu().numpy()

        #remove batch dimension
        output = output[0]

        #convert output to binary
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        return output

#Skeletonize class from skeletonize.py file that runs tile through skeletonization
import numpy as np
import skimage.morphology as morphology

class Skeletonize:
    def __init__(self, threshold):
        self.threshold = threshold

    def skeletonize(self, tile):
        #convert tile to binary
        tile[tile >= self.threshold] = 1
        tile[tile < self.threshold] = 0

        #skeletonize tile
        tile = morphology.skeletonize(tile)

        return tile
    



