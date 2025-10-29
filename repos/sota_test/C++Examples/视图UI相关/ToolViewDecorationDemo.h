#pragma once
/** @class
*  @brief   屏幕装饰范例
*  @author  北京构力科技有限公司
*  @date    2021/11/29
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2021/11/29
*  ------------------------------------------------------------
*  @note:  屏幕装饰不可选中、不可捕捉
*/


//装饰类
class ViewDecorationDemo:BIMBase::Core::BPViewDecoration
{

public:
	ViewDecorationDemo();
	~ViewDecorationDemo();

	static ViewDecorationDemo& get();
	void begin();
	void end();

	virtual bool          _drawDecoration(BPViewportR viewport);
};

//启动装饰类工具
class ToolLayoutViewDecorationDemo :public BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutViewDecorationDemo();
	~ToolLayoutViewDecorationDemo();


protected:
	virtual ::p3d::Utf8CP _getToolName() const { return "layoutDecorationDemo"; }
	virtual void _onPostInstall() override;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;
};

