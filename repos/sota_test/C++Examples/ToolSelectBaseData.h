#pragma once
/** @class
 *  @brief   二进制存取范例里取数据的选择工具（暂时不用）
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2022/4/26
 *  ------------------------------------------------------------
 *  @note:  -
 */
class ToolSelectBaseData : public BPEntitySelectSetTool
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolSelectBaseData();
	~ToolSelectBaseData();

protected:
	virtual Utf8CP _getToolName() const override { return "ToolSelectBaseData"; }
	virtual bool   _onDataButton(BPBaseButtonEventCP) override;
	virtual void   _onRestartTool() override;
	virtual void   _exitTool() override;
	virtual bool   _onResetButton(BPBaseButtonEventCP) override;
	virtual void   _onPostInstall() override;
	virtual UsesDragSelect      _allowDragSelect() override { return USES_DRAGSELECT_None; }
	virtual bool                _needPointForSelection()   override { return false; }
	virtual bool                _needPointForDynamics()  override { return false; }
	virtual bool                _needAcceptPoint()override { return true; }
	virtual p3d::StatusInt       _onEntityModify(BPEntityR el) override;
	virtual bool                _wantAdditionalLocate(BPBaseButtonEventCP ev)override { return true; }
	virtual BIMBase::Core::BPEntityPtr _buildLocateAgenda(BPPickDataCP path, BPBaseButtonEventCP ev) override;

	
};

