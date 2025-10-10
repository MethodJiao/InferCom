from pyp3d import *
# 定义参数化模型
class 弧(Component):
    # 定义各个参数及其默认值
    def __init__(self):
        Component.__init__(self)
        self['a轴长度'] = Attr(1000.0, obvious=True, readonly = False)
        self['b轴长度'] = Attr(300.0, obvious=True)
        self['弧'] = Attr(None, show=True)
        self['旋转角度'] = Attr(0,obvious = True)
        self['X'] = Attr(300.0, obvious=True)
        self['Y'] = Attr(300.0, obvious=True)
        self['Z'] = Attr(300.0, obvious=True)

        self.replace()
    @export
    # 开始写模型
    def replace(self):
        # 设置变量，同时调用参数(简化书写过程)
        L = self['a轴长度']
        W = self['b轴长度']
        x = self['X']
        y = self['Y']
        z = self['Z']
        Angle = self['旋转角度']
        # 绘制模型
        self['弧'] = translate(x,y,z) * rotation(math.pi/180 * Angle) * scale(L,W) * Arc(math.pi*2)
        # self['弧'] = Arc(Vec3(0,0,0),Vec3(100,100,100),Vec3(0,200,100))
        
# 输出模型
if __name__ == "__main__":
    FinalGeometry = 弧()
    place(FinalGeometry)
