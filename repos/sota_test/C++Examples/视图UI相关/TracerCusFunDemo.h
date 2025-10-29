#pragma once
//自定义追踪器说明：
//1.添加相关头文件详见stdafx.h追踪器相关,添加WDUi.lib
//2.继承wd::WDTracerCusFunBase，重写相关接口（接口含义见注释），实现自定义追踪器
//3.在工具中使用追踪器，详见ToolLayoutLineDemo，主要包括通过wd::TracerEvState定义追踪器;
//并在工具启动和左键点击时更新追踪器;在鼠标动态时对追踪器传入点;重写_OnTracerChanged接口
//获取追踪器的值直接布置构件


class TracerCusFunDemo : public Tracer::TracerCusFunBase
{
public:
	enum class eStateType
	{
		TRACER_DEMO = 90,  /**< 自定义点追踪器，为避免冲突，演示代码从90开始*/
	};
public:
	TracerCusFunDemo();
    virtual ~TracerCusFunDemo();

protected:
    /**
	 @brief		设置追踪器面板信息
	 @detail		初始化追踪器时，设置追踪器显示内容
	 @param[out]	ref	    追踪器面板数据
	 @return		无返回
	 */
    virtual void _OnPlaceCtrl(OUT Tracer::CtrlList& ref) override;

    /**
    @brief		更新追踪器面板信息
    @detail		鼠标移动时，通过赋值StartPt――ptNew 实时调用并更新追踪器面板值
    @param[in]	ptLast	起始点，参照点
    @param[in]	ptNew	鼠标当前点
    @param[out]	ref	    追踪器面板数据
    @return		无返回
    */
    virtual void _OnUpdateUi(OUT Tracer::CtrlList& ref, IN GePoint3dCR ptLast, IN GePoint3dCR ptNew) override; 
                                         
    /**
    @brief		修改追踪器面板数据后
    @detail		根据修改后的参数，计算关联参数和ptNew，更新追踪器显示数据并返回ptNew
    @param[in]	ptLast	起始点，参照点
    @param[out]	ref	    追踪器面板数据
    @param[out]	ptNew	鼠标当前点
    @return	    参数是否合法
    */
    virtual bool _OnSurePlace(OUT Tracer::CtrlList& ref, IN GePoint3dCR ptLast, OUT GePoint3d& ptNew) override;
};

