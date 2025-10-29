#pragma once
/** @class
 *  @brief   布置绝缘子串工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */
#include"InsulatorDemo.h"

class ToolLayoutInsulatorDemo : public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutInsulatorDemo();
	~ToolLayoutInsulatorDemo();

protected:
	//工具ID
	virtual Utf8CP _getToolName() const override { return "layoutInsulatorDemo"; }
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
	//向工程中增加立方体
	void __addInsulator(PModelId modelId);
	//单点布置时创建立方体数据
	void __createOnePtData(GePoint3d ptOri);
	
private:
	
	DemoObject::InsulatorDemo m_Insulator;

	/**联数*/
	int m_nN;
	/**单串绝缘子片数量*/
	int m_nN1;
	/**绝缘子单片连接高度*/
	double m_dH1;
	/**大伞裙半径*/
	double m_dR1;
	/**小伞裙半径*/
	double m_dR2;
	/**绝缘子串半径*/
	double m_dR;
	/**双串间距*/
	double m_dD;
	/**前端长度（构架端*/
	double m_dFL;
	/**后端长度（导线端）*/
	double m_dAL;
	/**连接导线分裂数*/
	int m_nLN;
};

