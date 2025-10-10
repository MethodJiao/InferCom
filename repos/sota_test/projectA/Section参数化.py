from pyp3d import *
# 放样体

class 放样体(Component):
    def __init__(self):
        Component.__init__(self)
        self['边长'] = Attr(1000.0, obvious=True)
        # self['宽'] = Attr(300.0, obvious=True)
        # self['高'] = Attr(500, obvious = True)
        self['放样体'] = Attr(None, show=True)

        self.replace()
    @export
    def replace(self):
        L = self['边长']
        # W = self['宽']
        # H = self['高']
        # 描述截面
        test_section = Section(Vec2(L/2,-L/2), scale(L/2) * Arc(0.5*pi), Vec2(-L/2,L/2), Vec2(-L/2,-L/2)).color(100/255,0/255,100/255,0.7)
        
        self['放样体'] = test_section

if __name__ == "__main__":
    FinalGeometry = 放样体()
    place(FinalGeometry)


