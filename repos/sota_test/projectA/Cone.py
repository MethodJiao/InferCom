from pyp3d import *
# 定义参数化模型
class 圆柱体(Component):
    # 定义各个参数及其默认值
    def __init__(self):
        Component.__init__(self)
        self['顶面半径'] = Attr(0.01, obvious = True)
        # obvious 属性的可见性 True时可见，False为不可见。默认为False
        # readonly 属性的只读性 True时不可调，为置灰状态，False为可调状态。默认为False
        self['底面半径'] = Attr(1000.0, obvious = True)
        self['高'] = Attr(5000, obvious = True)
        self['圆柱体'] = Attr(None, show = True)
        self.replace()
    @export
    # 模型造型
    def replace(self): 
        # 设置变量，同时调用参数(简化书写过程)
        Rt = self['顶面半径']
        Rb = self['底面半径']
        H = self['高']
        # 绘制模型
        TestCone = Cone(Vec3(0,0,0),Vec3(0,0,H),Rb,Rt).color(1,1,0,1)
        self['圆柱体'] = TestCone
# 输出模型
if __name__ == "__main__":
    FinalGeometry = 圆柱体()
    place(FinalGeometry)
