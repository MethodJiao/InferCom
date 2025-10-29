#pragma once
/**
* @class  	ToolPlanGraphDemo
* @brief  	建筑领域平面图功能范例
* @author   北京构力科技有限公司
* @date   	2023/03/30
* ------------------------------------------------------------
* 版本历史       注释                日期
* ------------------------------------------------------------
* @version v1.0  初始版本            2023/03/30
* ------------------------------------------------------------
* Note:
*/

class ToolPlanGraphDemo : public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool);

public:
	ToolPlanGraphDemo();

protected:
	virtual ~ToolPlanGraphDemo();
	virtual ::p3d::Utf8CP _getToolName() const;
	virtual void _onRestartTool() override;
	virtual bool _onDataButton(BPBaseButtonEventCP) override;
	virtual bool _onResetButton(BPBaseButtonEventCP) override;
	virtual bool _onModelMotion(BPBaseButtonEventCP ev)override;
	virtual void _onDynamicFrame(BPBaseButtonEventCP) override;
	virtual void _onPostInstall() override;

private:
	void __doCutting(int);
	//vecplane代表剖切面，0是xy平面，1是yz平面，2是xz平面
	void __createSectionBoxAndClipPlane(int vecPlane, GeRange3d range, GePlane3d& clipplane, GeTransform& sectionBox, GeTransform& transform);
	void __postTreatment();

private:
	int m_Num = -1;
	BPModelPtr m_ptrModelNew;
};