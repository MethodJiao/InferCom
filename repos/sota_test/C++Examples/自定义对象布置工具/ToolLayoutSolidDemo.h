#pragma once
/** @class  
 *  @brief   布置几何体工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/26
 *  ------------------------------------------------------------
 *  @note:  -  
 */

class ToolLayoutSolidDemo : public BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutSolidDemo();
	~ToolLayoutSolidDemo();

	enum CubeLayoutWay
	{
		OnePoint,//一点布置
		Draw,    //两点绘制
	};

protected:
	//工具ID
	virtual Utf8CP _getToolName() const override { return "layoutSolidDemo"; }
	//工具启动后响应函数
	virtual void _onPostInstall() override;
	//重启工具响应函数
	virtual void _onRestartTool() override;
	//点击鼠标左键响应函数
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	//点击鼠标右键响应函数
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	//动态函数
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	//鼠标移动响应函数，决定是否开启动态函数
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;
	//键盘按键响应函数
	virtual bool _onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown) override;

private:
	//向工程中增加几何体
	void __addSolid(PModelId modelId);
	BPPlacement cacuPlacement();
private:
	GeSolidBaseType m_eSolidType;
	std::vector<GePoint3d> m_vctPts;
	DemoObject::SolidDemo  m_Solid;
	int m_nRotStep;
};

