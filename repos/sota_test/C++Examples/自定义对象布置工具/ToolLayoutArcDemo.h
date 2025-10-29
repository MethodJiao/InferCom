#pragma once
/** @class
*  @brief   布置工具范例：两点画弧线，并进行弧长和角度标注
*  @author  北京构力
*  @date    2020/11/30
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/11/30
*  ------------------------------------------------------------
*  @note:  -
*/

class ToolLayoutArcDemo :public BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutArcDemo();
	~ToolLayoutArcDemo();

protected:
	virtual ::p3d::Utf8CP _getToolName() const { return "layoutArcDemo"; }
	virtual void _onPostInstall() override;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;
	virtual bool _onKeyTransition(bool isDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)override;

private:
	BPGraphicsPtr m_ptrGraphic;
	GePoint3d m_ptF;
	GePoint3d m_ptS;
	GePoint3d m_ptC;
	int m_nType;
	int m_nStep;
	
private:	
	void __createGraphic(GePoint3d endPoint, BPProjectP pProject, BPModelP pModel);
	//动态标注
	BPGraphicsPtr __dynamicDimension(GePoint3d ptE, BPProjectP pProject, BPModelP pModel);
};


