#pragma once
/** @class
 *  @brief   布置桥架工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */


namespace DemoObject
{
	class ToolLayoutIndependentBridge :public BIMBase::Core::BPPrimitiveTool
	{
		DefineSuper(BPPrimitiveTool)
	protected:
		//工具ID
		virtual Utf8CP _getToolName() const override { return "layoutIndependentBridgeDemo"; }
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
		IndependentBridgePtr __createIB(GePoint3d, GePoint3d);

	private:
		vector<GePoint3d> m_gPts;
	};
}
