import numpy as np

class ShDisplay(BaseProcessingObj):
    '''
    TODO not yet working
    '''
    def __init__(self, sh=None, pyr_style=False):
        super().__init__('sh_display', 'SH display')
        self._wsize = [600, 600]
        self._window = 21
        self._opened = False
        self._disp_factor = 0.0
        self._pyr_style = pyr_style

        if sh is not None:
            self._sh = sh

    def set_w(self):
        # Simulating the window opening based on size
        print(f"Opening window {self._window} with size {self._wsize}")

    def trigger(self, t):
        s = self._sh.out_i

        if s.generation_time == t:
            if not self._opened:
                self.set_w()
                self._opened = True

            print(f"Setting window: {self._window}")

            if self._pyr_style:
                img = self._sh.out_i.i
                img2 = np.zeros_like(img)
                dimx = self._sh.lenslet.dimx
                dimy = self._sh.lenslet.dimy
                np_sub = self._sh.subap_npx
                x2 = np_sub * dimx // 2
                y2 = np_sub * dimy // 2
                npx2 = np_sub // 2

                for y in range(dimy):
                    for x in range(dimx):
                        subap = img[x * np_sub:(x + 1) * np_sub, y * np_sub:(y + 1) * np_sub]
                        img2[x * npx2:(x + 1) * npx2, y * npx2:(y + 1) * npx2] = subap[:npx2, :npx2]
                        img2[x * npx2 + x2:(x + 1) * npx2 + x2, y * npx2:(y + 1) * npx2] = subap[npx2:, :npx2]
                        img2[x * npx2:(x + 1) * npx2, y * npx2 + y2:(y + 1) * npx2 + y2] = subap[:npx2, npx2:]
                        img2[x * npx2 + x2:(x + 1) * npx2 + x2, y * npx2 + y2:(y + 1) * npx2 + y2] = subap[npx2:, npx2:]

                self.image_show(img2)
            else:
                self.image_show(self._sh.out_i.i)

    def image_show(self, img):
        # Display image logic - Placeholder for actual display code
        print(f"Displaying image with shape: {img.shape}")

    def run_check(self, time_step):
        return obj_valid(self._sh)

    def cleanup(self):
        # Clean up resources if needed
        pass
