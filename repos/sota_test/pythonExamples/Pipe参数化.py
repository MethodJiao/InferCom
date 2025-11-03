from pyp3d import *
# 定义参数化模型
class 管道(Component):
    # 定义各个参数及其默认值
    def __init__(self):
        Component.__init__(self)
        self['内径'] = Attr(400, obvious = True)
        # obvious 属性的可见性 True时可见，False为不可见。默认为False
        # readonly 属性的只读性 True时不可调，为置灰状态，False为可调状态。默认为False
        self['外径'] = Attr(500.0, obvious = True)
        self