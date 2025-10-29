#pragma once
/** @class
*  @brief   布置工具范例：单点布球，并关联立方体
*  @author  北京构力
*  @date    2020/11/30
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/11/30
*  ------------------------------------------------------------
*  @note:  -
*/
class ToolLayoutBallTest :public BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutBallTest();
	~ToolLayoutBallTest();

protected:
	virtual ::p3d::Utf8CP _getToolName() const { return "LayoutBall"; }
	virtual void _onPostInstall() override;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;

private:
	BPGraphicsPtr m_ptrGraphic;
	GePoint3d m_ptC;
	TestObject::CubeTestPtr m_ptrCube;
};

