#pragma once
/** @class ToolArchClashDemo
*  @brief  建筑领域碰撞检查范例,选择要检查的构件后右键获取碰撞结果
*  @author 北京构力
*  @date   2023/03/30
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本                   2023/03/30
*  ------------------------------------------------------------
*  @note:  -
*/


#include "BPInteraction\BPEntitySelectSetTool.h"

class ToolArchClashDemo : public BPEntitySelectSetTool
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolArchClashDemo();
	~ToolArchClashDemo();

protected:
	virtual Utf8CP _getToolName() const override;
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

private:
	vector<BPEntityPtr> m_vcEEH;
};
