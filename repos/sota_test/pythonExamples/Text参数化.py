from pyp3d import *

class 文字(Component):
    def __init__(self):
        Component.__init__(self)
        self['文字内容'] = Attr('BIMBase参数化组件', obvious = True)
        self['文字'] = Attr(None, show = True)
        self.replace()
    @export
    def replace(self):
        test_text = self['文字内容']

        test_text_Final = Text(test_text,100,120)
        test_Line = Line(Vec3(0,0,0),Vec3(0,0,1))
        self['文字'] = test_text_Final
        # self['文字'] = Sweep(test_text_Final,test_Line).color(0,0,1,0.8)

if __name__ == "__main__":
    FinalGeometry = 文字() 
    place(FinalGeometry)
