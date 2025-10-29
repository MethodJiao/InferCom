#pragma once
#include "ModeArchWallDemo.h"	
/** @class
 *  @brief   布置建筑墙
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2023/03/30
 *  ------------------------------------------------------------
 *  @note:  -
 */

class ToolArchWallDemo : public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolArchWallDemo();
	~ToolArchWallDemo();

protected:
	//工具ID
	virtual Utf8CP _getToolName() const override { return "layoutCubeDemo"; }
	//工具启动后响应函数
	virtual void _onPostInstall() override;
	//重启工具响应函数
	virtual void _onRestartTool() override;
	//点击鼠标左键响应函数
	virtual bool _onDataButton(BIMBase::Core::BPBaseButtonEventCP) override;
	//点击鼠标右键响应函数
	virtual bool _onResetButton(BIMBase::Core::BPBaseButtonEventCP) override;
	//动态函数
	virtual void _onDynamicFrame(BIMBase::Core::BPBaseButtonEventCP) override;
	//鼠标移动响应函数，决定是否开启动态函数
	virtual bool _onModelMotion(BIMBase::Core::BPBaseButtonEventCP ev) override;
	//键盘按键响应函数
	virtual bool _onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown) override;


private:
	//向工程中增加立方体
	void __addCube(PModelId modelId);
	//单点布置时创建立方体数据
	void __createOnePtData(GePoint3d ptOri);
	//两点绘制时创建立方体数据
	void __createDrawData(GePoint3d ptOri, GePoint3d ptSecond);
private:
	ToolLayoutCubeDemo::CubeLayoutWay m_eLayoutWay;
	std::vector<GePoint3d> m_vctPts;
	DemoObject::CubeDemo m_Cube;
	int m_nHeight;
	int m_nWidth;
	int m_nLength;
};