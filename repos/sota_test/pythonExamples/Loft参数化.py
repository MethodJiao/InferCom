from pyp3d import *
# 放样体

class 放样体(Component):
    def __init__(self):
        Component.__init__(self)
        self['长'] = Attr(1000.0, obvious=True)
        self['宽'] = Attr(300.0, obvious=True)
        self['单层高'] = Attr(500, obvious = True)
        self['立方体'] = Attr(None, show=True)

        self.replace()
    @export
    def replace(self):
        L = self['长']
        W = self['宽']
        H = self['单层高']
        # 描述截面
        test_section1 = translate(-L/2,-W/2,0)*Section(Vec2(0,0), Vec2(L,0), Vec2(L,W), Vec2(0,W))
        # test_section2 = Section(scale(200) * Arc())
        # 放样
        # loft = Loft(test_section1, translate(0,0,L) * test_section1)
        loft = Loft(test_section1, translate(0,0,H) * test_section1,translate(0.2*L,0,2*H) * test_section1)
        
        self['立方体'] = loft

if __name__ == "__main__":
    FinalGeometry = 放样体()
    place(FinalGeometry)


