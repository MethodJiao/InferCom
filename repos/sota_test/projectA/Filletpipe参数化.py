from pyp3d import *
# 定义参数化模型
class 圆角管(Component):
    # 定义各个参数及其默认值
    def __init__(self):
        Component.__init__(self)
        self['管半径'] = Attr(10, obvious = True)
        # obvious 属性的可见性 True时可见，False为不可见。默认为False
        # readonly 属性的只读性 True时不可调，为置灰状态，False为可调状态。默认为False
        self['弯折半径'] = Attr(100.0, obvious = True)
        # self['高'] = Attr(5000, obvious = True)
        self['圆角管'] = Attr(None, show = True)
        self.replace()
    @export
    # 模型造型
    def replace(self): 
        # 设置变量，同时调用参数(简化书写过程)
        r = self['管半径']
        D = self['弯折半径']
        TestFilletPipe = FilletPipe(
            [Vec3(0,0,0),Vec3(1000,0,0),Vec3(1000,0,500),Vec3(2000,0,0),Vec3(2000,2000,0)],# 轨迹拐点
            [0,2*D,D,3*D,0],# 弯折半径，个数要与点的数量一致。第一个和最后一个点的弯折半径，一定是0。
            r # 管半径。弯折半径必须全部大于管半径。第一个和最后一个点除外。
            # 注意，弯折半径不要太大
        )
        # 绘制模型
        line = Line(Vec3(0,0,0),Vec3(1000,0,0))
        self['圆角管'] = TestFilletPipe
# 输出模型
if __name__ == "__main__":
    FinalGeometry = 圆角管()
    place(FinalGeometry)
