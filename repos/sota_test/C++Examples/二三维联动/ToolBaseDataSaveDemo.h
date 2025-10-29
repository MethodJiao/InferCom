#pragma once
/** @class
 *  @brief   二进制存取范例里存数据的工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */


class ToolBaseDataSaveDemo : public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BIMBase::Core::BPPrimitiveTool)
public:
	ToolBaseDataSaveDemo();
	~ToolBaseDataSaveDemo();

protected:
	virtual Utf8CP _getToolName() const override { return "ToolBaseDataSaveDemoDemo"; }
	virtual bool _onDataButton(BPBaseButtonEventCP ev) override;
	virtual void   _onRestartTool() override;
	virtual bool   _onResetButton(BPBaseButtonEventCP) override;
	virtual void   _onPostInstall() override;


private:
	
	pvector<DemoObject::BaseDataDemoP> m_BaseData;
};

/** @class
 *  @brief   二进制存取范例里取数据的工具
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */
class ToolBaseDataGet : public BIMBase::Core::BPPrimitiveTool
{
	DefineSuper(BIMBase::Core::BPPrimitiveTool)
public:
	ToolBaseDataGet();
	~ToolBaseDataGet();

protected:
	virtual Utf8CP _getToolName() const override { return "toolBaseDataGetDemo"; }
	virtual bool _onDataButton(BPBaseButtonEventCP ev) override;
	virtual void   _onRestartTool() override;
	virtual bool   _onResetButton(BPBaseButtonEventCP) override;
	virtual void   _onPostInstall() override;

	
};