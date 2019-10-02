## Script to collect training data for the CNN
## The flick coordinates with the highest SA relatned serve as indicator coordinates to retrieve positve gsv SA examples

from SA_predict import img_to_disk
from SA_geometry import *

#Get the high rated flickr coordinates
flickr_data = pd.read_csv("study_area_flickr_photos.csv", sep=";")
#Sort data
flickr_data = flickr_data.sort_values(by=" Grade", ascending=False)
#Get coords
coords = [i for i in zip(flickr_data[" Lat"], flickr_data[" Lon"])][:50]
def get_photos(coords, dest):
    #Create the surrounding bbox (see source)
    bbox = create_bbox(coords, polygonize=True)
    #Get data and model the polygon objects within the bbox
    polygons = extract_polygons(
        create_bbox(coords, polygonize=False)
    )
    #Aggregate the polygons into one big MultiPolygon object
    aggregated = aggregate_polygons(polygons, bbox)

    #Extract the *target polygon* which contains the facades (see source)
    target = extract_target_polygon(bbox, polygons)

    #Extract the *facades* from the target polygon (see source)
    facades = extract_facades(list(target.boundary.difference(bbox.boundary))) 
    #Segment the facades (if needed)
    facades = segment(facades, size=10)
    data = get_gsv_data(facades=facades, target=target,
                        photogeometric_fov=True,
                        as_dataframe=True)
    img_to_disk(images=list(data["photo"]), destination=dest, randomize=True)


for c in coords:
    get_photos(coord=c, dest="data")
