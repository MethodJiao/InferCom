from pyp3d import *
# 球体

class 球(Component):
    def __init__(self):
        Component.__init__(self)
        self['半径'] = Attr(1000.0, obvious=True, Combo = [1000,2000,3000])
        self['球'] = Attr(None, show=True)

        self.replace()
    @export
    def replace(self):
        r = self['半径']
        sphere = scale(r,r,r) * Sphere().color(0,1,0,1)
        # sphere = scale(r) * Sphere().color(0,1,0,1)
        self['球'] = sphere

if __name__ == "__main__":
    FinalGeometry = 球()
    place(FinalGeometry)

