from pyp3d import *

class 球(Component):
    def __init__(self):
        Component.__init__(self)
        self['半径'] = Attr(500, obvious = True)
        self['球'] = Attr(None, show = True)
        self['数量'] = Attr(10, obvious = True)
        self['球心距离'] = Attr(1500, obvious = True)
        
        self.replace()
    @export
    def replace(self):
        r = self['半径'] 

        test_sphere =scale(500) * Sphere()

        test_linspace = Array(test_sphere)
        L=self['球心距离']
        N=self['数量'] 
        # for 循环  线性排布
        for i in linspace(Vec3(0,0,0),Vec3(L*(N-1),0,0),N):
            test_linspace.append(translate(i))

        self['球'] = test_linspace


if __name__ == "__main__":
    FinalGeometry = 球()
    place(FinalGeometry)

