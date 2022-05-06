from htm.encoders.coordinate import CoordinateEncoder
from htm.encoders.grid_cell_encoder import GridCellEncoder
from htm.bindings.sdr import SDR
import numpy as np
from pyproj import Proj, transform
import math

# From http://spatialreference.org/ref/epsg/popular-visualisation-crs-mercator/
# This function now project from epsg:4326 (which is how we have long, lat coordinates) to epsg:3857 which is in meters. 
# Apparently not as accurate as some other systems but it is used for maps, I assume it is faster as a trade-off so I'll use it.
# I tested it here: http://epsg.io/map#srs=3857&x=1610569.987469&y=6464069.280250&z=17&layer=streets and it is still very accurate.
# Proj("epsg:3857")  # Spherical Mercator

class GeoSpatialEncoder(CoordinateEncoder):
    
    def __init__(self, scale, nearby_time, *args, **kwargs):
        """
        :param: scale size of the box for the encoding in meters, i.e. every scale meters a new SDR is created
        :param: nearby_time the number of seconds of travel that something is considered nearby 
        """
        super(GeoSpatialEncoder, self).__init__(*args, **kwargs)
        self.scale = scale
        self.nearby_time = nearby_time
        self.project = Proj("epsg:3857")
        self.min_radius = int(math.ceil((math.sqrt(self.w) - 1) / 2))
    
    def encode(self, input_data):
        """
        See `nupic.encoders.base.Encoder` for more information.
        :param: input_data (tuple) (longitude, latitude, speed)
        :returns: output SDR Stores encoded SDR in this numpy array
        """
        out = SDR(self.n)
        (longitude, latitude, speed) = input_data

        coord = self.coord_from_gps(longitude, latitude)
        radius = self.radius_from_speed(speed)

        super(GeoSpatialEncoder, self).encode((coord, radius), out)
        return out

    def coord_from_gps(self, longitude, latitude):
        """
        Returns coordinate for given GPS position.
        :param: longitude (float) Longitude of position
        :param: latitude (float) Latitude of position
        :returns: (numpy.array) 2D Integer Coordinate that the given GPS position
                            maps to
        """
        coordinate = np.array(self.project(longitude, latitude))
        coordinate = coordinate / self.scale # divide by scale and round with astype(int) to put coordinates into boxes of :scale: meter
        return coordinate.astype(int)

    def radius_from_speed(self, speed):
        """
        Returns radius for given speed.
        :param: speed (float) Speed (in km/h)
        :returns: (int) Radius for given speed
        """
        speed_in_ms = speed * 5/18 # 1000 / 3600 simplified
        coord_per_sec = speed_in_ms / self.scale
        radius = int(coord_per_sec * self.nearby_time)
        #return max(radius, self.min_radius)
        # By using max you give yourself a lower limit on how precise you are.
        # By simply adding, there is a continuous radius from even the lowest speed
        # the radius simply starts at min_radius and not at 0
        return self.min_radius + radius

    def get_output_size(self):
        return self.n

class GeoGridEncoder(GridCellEncoder):
    def __init__(self, rdse_speed, *args, **kwargs):
        super(GeoGridEncoder, self).__init__(*args, **kwargs)
        self.project = Proj("epsg:3857")
        self.rdse = rdse_speed
        self.custom_size = self.size + self.rdse.size
        
    def encode(self, input_data, grid_cells=None):
        """
        This function serves as a wrapper for GridCellEncoder where the longitude and latitude are automatically
        changed to a projection using meters, such that the periods of the grid cell encoder make sense
        :param: location a list [longitude, latitude]
        :param: input_data (tuple) (longitude, latitude, speed)
        """
        (longitude, latitude, speed) = input_data
        location = self.project(longitude, latitude)
        grid_sdr = super(GeoGridEncoder, self).encode(location, grid_cells)
        speed_sdr = self.rdse.encode(speed)
        out = SDR(self.custom_size)
        out.concatenate(grid_sdr, speed_sdr)
        return out

    def get_output_size(self):
        return self.custom_size
        
class MultiEncoder():
    def __init__(self, encoders):
        """
        :encoders: list of encoders to use in order: latitude, longitude, speed
        """
        self.encoders = encoders
        self.custom_size = sum([e.size for e in self.encoders])

    def encode(self, input_data):
        """
        This function serves as a wrapper for GridCellEncoder where the longitude and latitude are automatically
        changed to a projection using meters, such that the periods of the grid cell encoder make sense
        :param: input_data (tuple) (latitude, longitude, speed)
        """
        (latitude, longitude, speed) = input_data
        sdr_encoding = SDR(self.custom_size)
        sdr_encoding.concatenate([
            self.encoders[0].encode(latitude), 
            self.encoders[1].encode(longitude), 
            self.encoders[2].encode(speed)]
        )
        return sdr_encoding
    
    def get_output_size(self):
        return self.custom_size