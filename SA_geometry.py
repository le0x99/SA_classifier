import numpy as np
import pandas as pd
import sys
import requests
import warnings
from math import radians, sin, asin, cos, tan, atan, sqrt, atan2, degrees
import random
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, MultiLineString
from shapely.wkt import loads as to_geo
import shapely
from UliEngineering.Math.Coordinates import BoundingBox

plt.rcParams["figure.figsize"] = (12,12)

if 'ipykernel' in sys.modules:
	plt.rcParams["figure.dpi"] = 86
	plt.rcParams["figure.figsize"] = (10,5)
	from matplotlib import style
	style.use("seaborn")

def plot(obj:None, col:str, col2:str="grey", size:float=2.33) -> None:

    """Function to easier plot shapely objects,
Plottable : LineStrings, Polys, MultiPolys, tuples"""
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.axis("off")
            if type(obj) == shapely.geometry.polygon.Polygon:
                x, y = obj.exterior.xy    
                fig = plt.figure(1, figsize=(12,7.5), dpi=120)
                ax = fig.add_subplot(111)
                ax.plot(y, x, color=col, alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2)
                ax.fill(y, x, col2)
            elif type(obj) == shapely.geometry.multipolygon.MultiPolygon:
                for i in obj:
                    plot(i, col)
            elif type(obj) == list and type(obj[0]) == shapely.geometry.linestring.LineString:
                for i in obj:
                    x, y = i.xy
                    fig = plt.figure(1, figsize=(12,7.5), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.plot(y, x, color=col, alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
            elif type(obj) == shapely.geometry.linestring.LineString:
                plot([obj], col)
            elif type(obj) == tuple:
                x, y = obj[1], obj[0]
                plt.plot(x, y, "o", color=col, ms = 5)
            elif type(obj) == shapely.geometry.point.Point:
                plot(list(obj.coords)[0], col)
            elif type(obj) == list and type(obj[0]) == shapely.geometry.point.Point:
                for i in obj:
                    plot(i, col)
            elif type(obj) == list and type(obj[0]) == shapely.geometry.Polygon:
                for i in obj:
                    plot(i, col)   
            elif type(obj) == list and type(obj[0]) == tuple:
                for i in obj:
                    plot(i, col, size)
                
            
def create_bbox(p:tuple, polygonize=True) -> shapely.geometry.Polygon or list:
    """creates a bbox around a centroid within ~ 65m"""
    c = [
    (p[0]+0.0006729999999990355,p[1]),
    (p[0]-0.0006020000000006576,p[1]),
    (p[0], p[1] - 0.001235),
    (p[0], p[1] + 0.0010350000000000081) ]
    c = np.asarray(c) 
    bbox = BoundingBox(c)
    bbox = str(bbox).replace("BoundingBox","").replace("(","").replace(")","").split(",")   
    bbox[1], bbox[2] = bbox[2], bbox[1]
    bbox = [float(x) for x in bbox]
    return bbox if not polygonize else Polygon([ (bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])  ])

def destination_point(p:tuple, bearing:int,
                      d:float) -> tuple:
        lat1, lon1 = p
        lat1, lon1 = radians(lat1), radians(lon1)
        R = 6378.1                   #Radius of the Earth
        brng = radians(bearing) #convert degrees to radians
        d = d/1000                  #convert to m
        lat2 = asin( sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(brng))
        lon2 = lon1 + atan2(sin(brng)*sin(d/R)*cos(lat1),cos(d/R)-sin(lat1)*sin(lat2))
        return degrees(lat2), degrees(lon2)
       
def ellipse(p:tuple,
            d:float=4.) -> shapely.geometry.Polygon:
    return Polygon([destination_point(p, bearing,d) for bearing in range(0,361,10)])
            
 
def extract_polygons(bbox:list) -> dict:
    """Extract Polygon Objects from bbox area,
    returns a dictionary with ID as key and the corresponding polygon object as value"""
    d = dict()
    bbox = str(bbox).replace("[","").replace("]","")
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
[out:json][timeout:800];
(
  node["building"="yes"](bbox);
  way["building"="yes"](bbox);
  relation["building"="yes"](bbox);
  node["wall"="yes"](bbox);
  way["wall"="yes"](bbox);
  relation["wall"="yes"](bbox);
);

out skel geom;
>;
out skel geom;

""".replace("bbox",str(bbox).replace("[","").replace("]","")).replace("\n", "")
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    for element in data["elements"]:
        if element["type"]=="way":
            nodes = list()
            for coord in element["geometry"]:
                nodes.append((coord["lat"], coord["lon"]))
                
            d[element["id"]]=Polygon(nodes)
    return d

def aggregate_polygons(polygons:dict,
		       poly_bbox:shapely.geometry.Polygon,
		       intersect=False) -> shapely.geometry.MultiPolygon:
    """returns a MultiPolygon Object from all Buildings"""
    if intersect:
        return MultiPolygon([poly_bbox.intersection(polygons[i]) for i in polygons])
    else:
        return MultiPolygon([polygons[i] for i in polygons])



def extract_target_polygon(poly_bbox:shapely.geometry.Polygon,
			   polygons:dict, symmetric=False) -> shapely.geometry.MultiPolygon:
    """Creates the target polygon"""
    target = poly_bbox
    if symmetric:
        if type(polygons) ==dict:            
            for polygon in polygons:
                target = target.symmetric_difference(polygons[polygon])
        else:
            for polygon in polygons:
                target = target.symmetric_difference(polygon)
            
    else:
        if type(polygons)== dict: 
            for polygon in polygons:
                target = target - polygons[polygon]
        else:
            for polygon in polygons:
                target = target - polygon
        
        return target

   
def extract_streets(bbox:list,geom=True,
                    show_query=False,
                    include_footways=False) -> list:
    """Extracts the streets within the bbox as linestring objects,
    if packed==True it returns only a list of LineString objects (streets)"""
    d = dict() #every way has an id with 2 bounds
    bbox = str(bbox).replace("[","").replace("]","")
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
[out:json][timeout:800];
(way["highway"](bbox);>;);
out body geom;
""".replace("bbox",str(bbox).replace("[","").replace("]","")).replace("\n", "")
    overpass_query = overpass_query.replace("geom", "") if not geom else overpass_query
    response = requests.get(overpass_url, params={'data': overpass_query})
    print(overpass_query) if show_query else None
    data = response.json()
    minors = ['pedestrian',
              'track',
              'raceway',
              'footway',
              'path',
              'horseway',
              'steps',
              'sidewalk',
              'cycleway',
              'lane',
              'track',
              'construction']
    minors.remove("footway") if include_footways else None
    is_street = lambda way : way["tags"]["highway"] not in minors
    streets = [ element for element in data["elements"] if element["type"] == "way" and is_street(element) ]
    results = []
    for street in streets:
         p1 = (street["geometry"][0]["lat"], street["geometry"][0]["lon"])
         p2 = (street["geometry"][-1]["lat"], street["geometry"][-1]["lon"])
         results.append(LineString([p1, p2]))
    return results


def haversine(c1:tuple, c2:tuple) -> float:
    """returns the haversine distance in Meters"""
    lat1, lon1, lat2, lon2 = c1[0], c1[1], c2[0], c2[1]
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r * 1000


def segment(line_list:list, size:int=10) -> list:
    """Takes a list of lines and segments them (if needed) """
    l1 = len(line_list)
    for line in line_list:   
        if length(line) > size:
            line_list.append(LineString([list(line.coords)[0], list(line.centroid.coords)[0]]))
            line_list.append(LineString([list(line.centroid.coords)[0], list(line.coords)[1]]))            
            line_list.remove(line)
    l2 = len(line_list)
    
    return line_list if l1==l2 else segment(line_list)


def gsv_point(facade:shapely.geometry.LineString,
              target:shapely.geometry.MultiPolygon=None,
              bbox:shapely.geometry.Polygon=None,
              radius:int=15, meta_only=False,
              include_outer=False) -> shapely.geometry.Point or dict :
    """locates the closest gsv point and returns its (lat, lon) as a Point Object
         Checks if the located gsv-point is within the corresponding
         area of the target polygon which belongs to the input facade"""
    p = list(facade.centroid.coords)[0] if type(facade) == shapely.geometry.LineString else facade
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&location="+str(p[0])+"%2C"+str(p[1])+"&source=outdoor&radius="+str(radius)+"&pitch=12&key=AIzaSyCrjQChUxWzzcsRQt0SFeomIC0jN5vaDBo"
    response = requests.get(url)
    data = response.json()
    if meta_only:
        return data
    if data["status"] == "OK" and "Google" in data["copyright"]:
        gsv_point = Point((data["location"]["lat"], data["location"]["lng"]))
        if not gsv_point.within(bbox):
            return
        if type(target) == shapely.geometry.multipolygon.MultiPolygon:        
            for t in target:
                if facade.touches(t.boundary) or  facade.within(t.boundary):
                    if gsv_point.within(t) or gsv_point.touches(t):
                        return gsv_point
        else: return gsv_point
    else: return
    
    
        
def angle(facade:shapely.geometry.LineString) -> float:

    if gsv_point(facade) != None:
        gsv_p = list(gsv_point(facade).coords)[0]
        line1 = LineString([gsv_p, list(facade.coords)[0]])
        line2 = LineString([gsv_p, list(facade.coords)[1]])
        slope_1 = (list(line1.coords)[0][1]-list(line1.coords)[1][1]) / (list(line1.coords)[0][0]-list(line1.coords)[1][0])
        slope_2 = (list(line2.coords)[0][1]-list(line2.coords)[1][1]) / (list(line2.coords)[0][0]-list(line2.coords)[1][0])
        tan_alpha = abs((slope_1 - slope_2 ) / (1+slope_1*slope_2))
        return math.degrees(math.atan(tan_alpha)) 
  

def triangle(facade:shapely.geometry.LineString) -> shapely.geometry.Polygon:
    """Creates a triangle with the points of the facade and the closest gsv point"""
    if gsv_point(facade) != None: 
        return Polygon([list(facade.coords)[0], list(facade.coords)[1], list(gsv_point(facade).coords)[0]] )


def gsv_dist(facade:shapely.geometry.LineString) -> float:
    """returns distance between facade centroid and corresponding gsv point"""
    return haversine(list(gsv_point(facade).coords)[0], list(facade.centroid.coords)[0])
 
def length(facade:shapely.geometry.LineString) -> float:
    """returns length in meters"""
    return haversine(list(facade.coords)[0], list(facade.coords)[1])


def match(facade:shapely.geometry.LineString, polygons:list, poly_bbox:shapely.geometry.Polygon) -> str:
    """Exact, but extremely inefficient calculation on which facade belongs to which osm item"""
    d = dict()
    d2 = dict()
    for poly in polygons:
        d[poly] = list()
        for line in segment(extract_facades2(list(target.boundary.difference(poly_bbox.boundary))), size = 1):
            d[poly].append(line.centroid.distance(facade.centroid))
    for key in d:
        d2[key] = min(d[key])
    for key2 in d2:
        if d2[key2] == min(d2.values()):
            return key2
  

def match_0(facade:shapely.geometry.LineString) -> shapely.geometry.Polygon:
    """matches a facade/segment to the corresponding osm building VER 000"""
    for polygon in polygons:
        for line in extract_facades([polygons[polygon]]):
            if facade.within(line):
                return polygon  
            
def extract_facades(boundary_list:list) -> list:
    l = list()
    for facade in boundary_list:
        coords = list(facade.coords)
        for i in range(0, len(coords)-1):
            l.append(LineString([coords[i], coords[i+1]]))
    return l 

def get_gsv_data(facades:list, target:shapely.geometry.MultiPolygon,
		 photogeometric_fov=True, as_dataframe=False) -> pd.DataFrame or dict:
    centroid = lambda x : list(x.centroid.coords)[0]
    length = lambda x : haversine(list(x.coords)[0], list(x.coords)[1])
    data = [ obs for obs in [{"facade" : facade,
                          "gsv_point" : gsv_point(facade=facade, target=target, radius=15)
                         } for facade in facades ] if obs["gsv_point"] != None ]
    for i in data:
        f, g = i["facade"], i["gsv_point"]
        i["dist"] = haversine(centroid(f), centroid(g))
        i["length"] = haversine(list(f.coords)[0], list(f.coords)[1])
        i["fov"] = 2 * atan(i["length"]/(2*i["dist"])) * 100 if photogeometric_fov else 90
        i["photo"] = "https://maps.googleapis.com/maps/api/streetview?size=640x640&location="+str(centroid(f)[0])+"%2C"+str(centroid(f)[1])+"&source=outdoor&radius=15&pitch=10&fov="+str(i["fov"])+"&key=AIzaSyCrjQChUxWzzcsRQt0SFeomIC0jN5vaDBo"
    return data if not as_dataframe else pd.DataFrame(data)
