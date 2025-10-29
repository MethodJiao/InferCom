#pragma once
/** @class
*  @brief   演示圆管工具
*  @author  北京构力科技有限公司
*  @date    2022/4/19
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2022/4/19
*  ------------------------------------------------------------
*  @note: 存的是世界坐标系下的点，一般情况下造型存局部坐标系下的点，具体可参考cubeDemo
*/


namespace DemoObject
{

	class ToolLayoutExampleTubeDemo :public BIMBase::Core::BPPrimitiveTool
	{
		DefineSuper(BPPrimitiveTool)
	public:
		ToolLayoutExampleTubeDemo();
		~ToolLayoutExampleTubeDemo();
	protected:
		//工具ID
		virtual Utf8CP _getToolName() const override { return "LayoutExaTube"; }
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
	private:
		pvector<GePoint3d> m_gPts;

	};

}