#pragma once
/** @class
*  @brief   布置工具范例：两点画线
*  @author  北京构力
*  @date    2020/11/30
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/11/30
*  ------------------------------------------------------------
*  @note:  -
*/

class ToolLayoutLineDemo :public BPPrimitiveTool
						, public Tracer::ITracerEvent
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutLineDemo();
	~ToolLayoutLineDemo();


protected:
	virtual ::p3d::Utf8CP _getToolName() const { return "layoutLineDemo"; }
	virtual void _onPostInstall() override;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;
	virtual void _onUserInput(const wchar_t* str, BPBaseButtonEventCP ev) override;

protected:
	//追踪器事件范例
	virtual void _onTracerTabEvent(const Tracer::TracerInfo& info) override;
	virtual void _onTracerEditedChanged(const Tracer::TracerInfo& info, int nSelId) override;
	virtual void _onTracerSureBtnClick(const Tracer::TracerInfo& info) override;
	virtual void _onTracerCancelBtnClick() override;
	virtual void _onTracerCoordBtnClick(const Tracer::TracerInfo& info) override;
	virtual void _onTracerEnterEdit() override;
	virtual void _onTracerExitEdit()override;

private:
	//追踪器定义
	void __setTracer();
	void __createGraphic(bool beDynamic = true,GePoint3d endPoint = GePoint3d::createByZero());

	//动态标注
	BPGraphicsPtr __dynamicDimension(GePoint3d ptFirst, GePoint3d ptSecond);
	//连续标注
	BPGraphicsPtr __dynamicContinueDimension(bool beDynamic = true,GePoint3d ptDynamic = GePoint3d::createByZero());
	//提示文字
	void __sendMessage(const CString& msg);
private:
	pvector<GePoint3d> m_vctPoint;
	vector<GePoint3d> m_vctLaundry;
	int m_nStep;
	BPGraphicsPtr m_ptrGraphic;
	GePoint3d m_ptC;
	Tracer::TracerEvState m_tracerEv;
	int m_nInputWay;
	BIMBase::BPEntityId m_EntityIdSave;
};

