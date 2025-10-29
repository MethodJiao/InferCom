#pragma once
//自定义追踪器说明：
//1.添加相关头文件详见stdafx.h追踪器相关,添加WDUi.lib
//2.继承wd::WDTracerCusFunBase，重写相关接口（接口含义见注释），实现自定义追踪器
//3.在工具中使用追踪器，详见ToolLayoutLineTest，主要包括通过wd::TracerEvState定义追踪器;
//并在工具启动和左键点击时更新追踪器;在鼠标动态时对追踪器传入点;重写_OnTracerChanged接口
//获取追踪器的值直接布置构件


class BPTracerCusFun : public wd::WDTracerCusFunBase
{
public:
	enum eCusTracerType
	{
		WORLD,   /*世界坐标*/
		DISTANCE, /*距离、角度*/
	};
public:
	BPTracerCusFun();
	~BPTracerCusFun();

protected:
	//根据鼠标位置刷新追踪器面板上显示的值
	virtual void _OnUpdateUi(wd::CtrlList& ref, GePoint3dCR ptLast, GePoint3dCR ptNew);
	//定义追踪器面板
	virtual void _OnPlaceCtrl(wd::CtrlList& ref);
	//如果在追踪器输入值，根据输入的值ref计算鼠标位置ptNew返回
	virtual void _OnSurePlace(wd::CtrlList& ref, GePoint3dCR ptLast, GePoint3d& ptNew);
};

