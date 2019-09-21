from tqdm import tqdm
import shapely
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from SA_geometry import *
from SA_predict import to_img, predict_img
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)

class Environment(object):
    def __init__(self, coords:tuple):
        print("creating 2d environment...")
        self.init_coords = coords
        self.sa_facades = list()
        self.intersection_points = list()
        self.headinglines = list()
        self.ellipses = list()
        self.interlines = list()
        #Create the surrounding bbox 
        self.bbox = create_bbox(coords,
                                polygonize=True)
        #Get data and model the polygon objects within the bbox
        self.polygons = extract_polygons(
            create_bbox(coords,
                        polygonize=False))
        #Aggregate the polygons into one big MultiPolygon object (in order to plot it)
        self.aggregated = aggregate_polygons(self.polygons, self.bbox)
        #Extract the *target polygon* which contains the facades (see source)
        self.target = extract_target_polygon(self.bbox, self.polygons)
        #Extract the *facades* from the target polygon (see source)
        #Segment the facades (if needed)
        self.facades = segment(extract_facades(list(self.target.boundary.difference(self.bbox.boundary))),
                               size=10)
        #plot(bbox, "black"), plot(aggregated, "blue"), plot(facades, "orange"), plot(coords, "green", size=7.33)
        #Add streets
        self.streets = extract_streets(create_bbox(coords,
                                                   polygonize=False),
                                       include_footways=True)
        #Create array of gsv_point coords
        gsv_points = [ gsv_point(f, self.target, self.bbox) for f in self.facades ]
        self.gsv_points = [ i for i in gsv_points if i != None ]
        
    def evaluate(self, model,gsv_key:str, radius:float=5,
                 dirrlength:float=40):
        url = lambda lat, lon, heading : "https://maps.googleapis.com/maps/api/streetview?size=640x640&location="+str(lat)+"%2C"+str(lon)+"&source=outdoor&radius=1&heading="+str(heading)+"&pitch=10&key={}".format(gsv_key)
        make_line = lambda p_, l, h : LineString([p_, destination_point(p_, h, l)]) 
        tuples = [ (_.coords.xy[0][0], _.coords.xy[1][0]) for _ in self.gsv_points ]
        self.results = list()
        print("Interpreting images...")
        print("Total Images to interpret: {}".format(len(range(0,361,30)) * len(self.gsv_points)))
        for point in tqdm(tuples):
                for heading in range(0,361,30):
                    img_url = url(point[0], point[1], heading ) if model != None else None
                    #print(img_url)
                    img = to_img(url=img_url, size=(224, 224)) if model != None else None
                    pred = predict_img(model=model, img=img) if model != None else random.choice([1] + [0]*110)
                    if pred >= .99:
                        self.results.append( {        #"image" : img,
                                                  "point" : point,
                                                  "ellipse" : ellipse(point, radius),
                                                  "url" : img_url,
                                                  "heading" : heading,
                                                  })
        print("Extracting relevant facades...")
        for res in self.results:
            res["inter"] = []
            res["headinglines"] = []
            res["facades"] = []
            res["interpoints"] = []
            res["points"] = []
            self.ellipses.append(res["ellipse"])
            for street in self.streets:
                if res["ellipse"].intersects(street):
                    inter = res["ellipse"].intersection(street)
                    if length(inter) >= 8.:
                        inter = quad_points(inter)
                    elif length(inter) >= 5.:
                        inter = double_points(inter)
                    else:
                        inter = LineString([inter.coords[:][0],
                                           inter.centroid.coords[:][0],
                                           inter.coords[:][1]])
                    res["inter"].append(inter) 
                    self.interlines.append(inter)
                    for p in inter.coords[:]:
                        dirrline = make_line(p, dirrlength, res["heading"] )
                        res["headinglines"].append(dirrline)
                        self.headinglines.append(dirrline)

            for hline in res["headinglines"]:
                candidates = []
                for f in self.facades:
                    if hline.intersects(f):
                        intersection_point = hline.intersection(f)
                        candidates.append({"facade" : f,
                                           "distance" : haversine(intersection_point.coords[:][0],
                                                                                hline.coords[:][0] ),
                                          "point" : intersection_point})
                for candidate in candidates:
                    if candidate["distance"] == min([i["distance"] for i in candidates]):
                        res["facades"].append(candidate["facade"])
                        self.sa_facades.append(candidate["facade"])
                        res["interpoints"].append(candidate["point"])
                        self.intersection_points.append(candidate["point"])
                        
