#pragma once
/**@class
*  @brief  二维曲线求交
*  @author 北京构力科技有限公司
*  @date  2022/5/16
*-------------------------------------------
*版本历史        注释        日期
*-------------------------------------------
*  @version v1.0 初始版本  2022/5/16   
*-------------------------------------------
*  @note:
*/

class ToolCurveIntersentionDemo :public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BIMBase::Core::BPPrimitiveTool)
public:
	ToolCurveIntersentionDemo();
	~ToolCurveIntersentionDemo();


protected:
	virtual ::p3d::Utf8CP _getToolName() const { return "curveIntersentionDemo"; }
	virtual void _onPostInstall() override;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev) override;

private:
	GeCurveArrayPtr __createCurve();
	void __calculateIntersention(GeCurveArrayPtr ptrCurveA, GeCurveArrayPtr ptrCurveB);

private:
	pvector<GePoint3d> m_vctPts;
};

