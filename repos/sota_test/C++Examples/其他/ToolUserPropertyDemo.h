#pragma once
/** @class ToolUserPropertyDemo
*  @brief  自定义属性范例，包括获取自定义属性，添加自定义属性
*  @author 北京构力
*  @date   2021/04/08
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本                   2021/04/08
*  ------------------------------------------------------------
*  @note:  -
*/


#include "BPInteraction\BPEntitySelectSetTool.h"

class ToolUserPropertyDemo : public BPEntitySelectSetTool
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolUserPropertyDemo();
	~ToolUserPropertyDemo();

protected:
	virtual Utf8CP _getToolName() const override { return "userPropertyDemo"; }
	virtual bool   _onDataButton(BPBaseButtonEventCP) override;
	virtual void   _onRestartTool() override;
	virtual void   _exitTool() override;
	virtual bool   _onResetButton(BPBaseButtonEventCP) override;
	virtual void   _onPostInstall() override;
	virtual UsesDragSelect      _allowDragSelect() override { return USES_DRAGSELECT_Box; }
	virtual bool                _needPointForSelection()   override { return false; }
	virtual bool                _needPointForDynamics()  override { return false; }
	virtual bool                _needAcceptPoint()override { return true; }
	virtual p3d::StatusInt       _onEntityModify(BPEntityR el) override;
	virtual bool                _wantAdditionalLocate(BPBaseButtonEventCP ev)override { return true; }
	virtual BPEntityPtr _buildLocateAgenda(BPPickDataCP pickData, BPBaseButtonEventCP ev);

private:
	BPEntityPtr m_curEEH;
	BPGraphicElementPtr m_curObj;
};
