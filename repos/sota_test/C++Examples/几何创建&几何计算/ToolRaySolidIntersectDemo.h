#pragma once
/**@class
*  @brief  射线与实体求交
*  @author 北京构力科技有限公司
*  @date  2022/5/14
*-------------------------------------------
*版本历史        注释        日期
*-------------------------------------------
*  @version v1.0 初始版本  2022/5/14   
*-------------------------------------------
*  @note:
*/

class ToolRaySolidIntersectDemo :public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BIMBase::Core::BPPrimitiveTool)
public:
	ToolRaySolidIntersectDemo();
	~ToolRaySolidIntersectDemo();


protected:
	virtual ::p3d::Utf8CP _getToolName() const { return "raySolidIntersectDemo"; }
	virtual void _onPostInstall() override;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;

private:
	//绘制求交的立方体
	IGeSolidBasePtr __getSolid();
	void __drawSphere(GePoint3d);

private:
	std::vector<GePoint3d> m_ptClick;
};

