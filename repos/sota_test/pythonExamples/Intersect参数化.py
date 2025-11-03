from pyp3d import *
# 融合（布尔操作并与布尔操作减

class 布尔操作(Component):
    def __init__(self):
        Component.__init__(self)
        self['长'] = Attr(1000.0, obvious=True)
        self['宽'] = Attr(300.0, obvious=True)
        self['高'] = Attr(500, obvious = True)
        self['布尔和组合'] = Attr(None, show=True)

        self.replace()
    @export
    def replace(self):
        L = self['长']
        W = self['宽']
        H = self['高']

        TestCube_a = translation(100,100,0) * scale(L,W,H) * Cube().color(1,0,0,1)
        TestCube_b = translation(0,0,0) * scale(L,W,H) * Cube().color(0,0,1,1)
# 
        # self['布尔和组合'] = TestCube_a
        # self['布尔和组合'] = TestCube_b
        # self['布尔和组合'] = TestCube_a+TestCube_b
        self['布尔和组合'] = TestCube_b-TestCube_a
        # self['布尔和组合'] = Intersect(TestCube_a,TestCube_b)
        # self['布尔和组合'] = Combine(TestCube_a,TestCube_b)

if __name__ == "__main__":
    FinalGeometry = 布尔操作()
    place(FinalGeometry)
