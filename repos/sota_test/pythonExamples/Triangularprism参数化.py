from pyp3d import *
# 定义参数化模型
class 三角柱(Component):
    # 定义各个参数及其默认值
    def __init__(self):
        Component.__init__(self)
        self['长'] = Attr(1000, obvious = True, combo = [500,1000,2000,3000,4000])
        # obvious 属性的可见性 True时可见，False为不可见。默认为False
        # readonly 属性的只读性 True时不可调，为置灰状态，False为可调状态。默认为False
        self['宽']