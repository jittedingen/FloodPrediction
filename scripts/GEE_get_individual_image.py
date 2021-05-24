import ee
import ee.mapclient
from geetools import batch
import os
from rasterstats import zonal_stats

output_dir='.'
country = 'Uganda'


ee.Initialize()

if output_dir == '.':
    output_dir = '../' + country.lower() + '/input/water_surface'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# get shapefile of catchments, mapt it to a feature collection
#shapefile = 'users/jacopomargutti/catchment_' + country.lower()
#fe_col = ee.FeatureCollection(shapefile)
shapefile = '../' + country.lower() + '/input/Admin/uga_admbnda_adm1_UBOS_v2.shp'
bounding_box = ee.Geometry.Rectangle([29.5794661801, -1.44332244223, 35.03599, 4.24988494736]) #Uganda
#bounding_box = ee.Geometry.Rectangle([33.8935689697, -4.67677, 41.8550830926, 5.506]) #Kenya

collection = "JRC/GSW1_2/GlobalSurfaceWater"
#collection = "CGIAR/SRTM90_V4"
collection_dir = collection.replace('/', '_')
file_name = output_dir + '/'

image_scale = 30 #set this equal to the resolution of the image on GEE in meters
image = ee.Image(collection)

print('downloading', file_name)
batch.image.toLocal(image,
                    file_name, scale = image_scale,
                    region=bounding_box)

zs = zonal_stats(shapefile, output_dir + '/' + "GlobalSurfaceWater.occurrence.tif",
            stats="mean median")



################# ORIGINAL CODE BELOW ############

ee.Initialize()

output_dir = 'data'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# define bounding box of the Philippines
bounding_box = ee.Geometry.Rectangle([114.0952145, 4.2158064, 126.8072562, 21.3217806])


collection = 'CGIAR/SRTM90_V4'

collection_dir = collection.replace('/', '_')
file_name = output_dir + '/' + collection_dir

image_scale = 1000
image_agg = ee.Image(collection)

print('downloading', file_name)
batch.image.toLocal(image_agg,
                    file_name,
                    scale=image_scale,
                    region=bounding_box)