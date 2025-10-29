#pragma once
/** @class
 *  @brief   布置路堤工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */

class ToolLayoutEmbankmentDemo : public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BPPrimitiveTool)
public:
	ToolLayoutEmbankmentDemo();
	~ToolLayoutEmbankmentDemo();

protected:
	//工具ID
	virtual Utf8CP _getToolName() const override { return "layoutEmbankmentDemo"; }
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
	void __addEmbankment(PModelId modelId);
	//单点布置时创建立方体数据
	void __createOnePtData(GePoint3d ptOri);
private:
	DemoObject::EmbankmentDemo m_Embankment;
	int m_nTopWidth;
	int m_nGravelThickness;
	int m_nPackingThickness;
	int m_nLenght;
	double m_dSlop;
};

