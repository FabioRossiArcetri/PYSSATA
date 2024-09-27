from pyssata.base_data_obj import BaseDataObj

class Lenslet(BaseDataObj):
    def __init__(self, n_lenses=1, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._n_lenses = n_lenses
        self._lenses = []

        if n_lenses > 1:
            x, y = self.make_xy(n_lenses, 1.0)
        else:
            x = [0.0]
            y = [0.0]
        
        subap_size = 2.0 / n_lenses

        for i in range(n_lenses):
            row = []
            for j in range(n_lenses):
                row.append([x[i, j], y[i, j], subap_size])
            self._lenses.append(row)

    @property
    def n_lenses(self):
        return self._n_lenses

    @property
    def dimx(self):
        return len(self._lenses)

    @property
    def dimy(self):
        return len(self._lenses[0]) if self._lenses else 0

    def get(self, x, y):
        """Returns the subaperture information at (x, y)"""
        return self._lenses[x][y]

    def save(self, filename, hdr):
        """Saves the lenslet data to a file with the header information"""
        hdr['VERSION'] = 1
        self.base_data_obj.save(filename, hdr)
        self.xp.save(filename, self.xp.array(self._lenses))

    def read(self, filename, hdr, exten=0):
        """Reads lenslet data from a file and updates object state"""
        self.base_data_obj.read(filename, hdr, exten)
        self._lenses = self.xp.load(filename, allow_pickle=True).tolist()
        exten += 1

    @classmethod
    def restore(cls, filename):
        """Restores a lenslet object from a file"""
        data = self.xp.load(filename, allow_pickle=True)
        version = int(data['VERSION'])
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        p = cls()
        p.read(filename, hdr={})
        return p

    def cleanup(self):
        """Clean up the lenslet data"""
        del self._lenses
        self.base_data_obj.cleanup()

    @staticmethod
    def make_xy(n_lenses, scale):
        """Simulates the make_xy function to generate the lenslet grid"""
        xy = self.xp.linspace(-1, 1, n_lenses)
        x, y = self.xp.meshgrid(xy, xy)
        return x * scale, y * scale

    def revision_track(self):
        """Returns the revision number of the lenslet data"""
        return '$Rev$'

# Base class for simulation (assuming it's implemented elsewhere)
class BaseDataObj:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def save(self, filename, hdr):
        # Placeholder for saving the base object data
        pass

    def read(self, filename, hdr, exten):
        # Placeholder for reading the base object data
        pass

    def cleanup(self):
        # Placeholder for cleanup logic
        pass
