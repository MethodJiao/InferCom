from pyp3d import *

class 多段线(Component):
    def __init__(self):
        Component.__init__(self)
        self['长'] = Attr(1000.0, obvious=True, Group = 'AAA', description = 'testlength')
        self['半径'] = Attr(300.0, obvious=True, Group = 'AAA', description = 'testheight')
        # self['高'] = Attr(500, obvious = True)
        self['多段线'] = Attr(None, show=True)
        # self['X'] = Attr(300.0, obvious=True)
        # self['Y'] = Attr(300.0, obvious=True)
        # self['Z'] = Attr(300.0, obvious=True)

        self.replace()
    @export
    def replace(self):
        L = self['长']
        R = self['半径']
        # H = self['高']
        # x = self['X']
        # y = self['Y']
        # z = self['Z']
        
        line1 = Line(Vec2(0,0), Vec2(L,0), translate(L-R,0) * scale(R) * Arc(0.5*pi)).color(0,1,0,0.5)
        line2 = Line(Vec3(100,-100,0), scale(50) * Arc(0.5*pi), Vec3(-100,100,100), Vec3(-100,-100,200)).color(0,1,0,0.5)
        line3 = Line(Vec3(100,-100,0), Vec3(100,100,50), Vec3(-100,100,100), Vec3(-100,-100,200)).color(0,1,0,0.5)

        self['多段线'] = line1
        # self['多段线'] = line2
        # self['多段线'] = line3

if __name__ == "__main__":
    FinalGeometry = 多段线()
    place(FinalGeometry)
