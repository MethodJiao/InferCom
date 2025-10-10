from pyp3d import *
# 扫掠体

class 扫掠(Component):
    def __init__(self):
        Component.__init__(self)
        self['角度'] = Attr(90.0, obvious=True)
        self['边长'] = Attr(500.0, obvious=True)
        self['扫掠体'] = Attr(None, show=True)

        self.replace()
    @export
    def replace(self):
        r = self['角度']
        L = self['边长']
        sectionOut = rotate(Vec3(1,0,0), 0.5*pi) * Section(Vec2(0,0), Vec2(L,0), Vec2(L,L), Vec2(0,L))
        section_1 = translation(200,0,0) * rotate(Vec3(1,0,0), 0.5*pi) * Section(Vec2(80,80), Vec2(60,80), Vec2(60,60), Vec2(80,60))
        section_2 = translation(200,0,0) * rotate(Vec3(1,0,0), 0.5*pi) * Section(Vec2(50,50), Vec2(20,50), Vec2(20,20), Vec2(50,20))
        testarc = Arc(Vec2(0,0),Vec2(L,L),Vec2(0,L*2))
        line = Line(Arc(r*pi/180))
        line2 = Line(Vec2(100,-100), scale(50) * Arc(0.5*pi), Vec2(-100,100), Vec2(-100,-100))
        sweep = scale(500) * Sweep(sectionOut, line) # 
        self['扫掠体'] = sweep

if __name__ == "__main__":
    FinalGeometry = 扫掠()
    place(FinalGeometry)

