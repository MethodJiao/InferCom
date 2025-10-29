#pragma once
/** @class ToolSelectDemo
*  @brief  选择工具范例
*  @author 北京构力
*  @date   2021/04/08
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本                   2021/04/08
*  ------------------------------------------------------------
*  @note:  -
*/

class ToolSelectDemo : public BPEntitySelectSetTool
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolSelectDemo();
	~ToolSelectDemo();

protected:
	virtual Utf8CP _getToolName() const override { return "selectToolDemo"; }
	virtual void   _setupAndPromptForNextAction() override;
	virtual bool   _onDataButton( BPBaseButtonEventCP) override;
	virtual void   _onRestartTool() override;
	virtual bool   _onResetButton( BPBaseButtonEventCP) override;
	virtual void   _onDynamicFrame( BPBaseButtonEventCP) override;
	virtual void   _onPostInstall() override;
	virtual bool   _onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown) override;
	virtual UsesDragSelect      _allowDragSelect() override { return USES_DRAGSELECT_Box; }
	virtual bool                _needPointForSelection()   override { return false; }
	virtual bool                _needPointForDynamics()  override { return false; }
	virtual bool                _needAcceptPoint()override { return true; }
	virtual bool                _wantAdditionalLocate( BPBaseButtonEventCP ev)override { return true; }
	virtual bool                _onModifierKeyTransition(bool wentDown, int key)override;
};
