#pragma once
/** @class ToolReNumber
*  @brief  重新编号
*  @author 北京构力
*  @date   2021/04/08
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本                   2021/04/08
*  ------------------------------------------------------------
*  @note:  -
*/

//#include "BIMBase\P3DInteraction\P3DElementSelectSetTool.h"

class ToolReNumber : public ::BPEntitySelectSetTool//, virtual public ::PBBim::PBUIToWorkMessageEvent
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolReNumber();
	~ToolReNumber();

protected:
	virtual Utf8CP _getToolName() const override { return "TestForSelectTool"; }
	virtual void   _setupAndPromptForNextAction() override;
	//virtual bool   _IsValidElement(ElementHandleCR element) override;
	virtual bool   _onDataButton( BPBaseButtonEventCP) override;
	virtual void   _onRestartTool() override;
	virtual bool   _onResetButton( BPBaseButtonEventCP) override;
	virtual void   _onDynamicFrame( BPBaseButtonEventCP) override;
	//virtual bool   _OnInstall() override;
	virtual void   _onPostInstall() override;
	//virtual EditElementHandleP PB_BuildLocateAgenda(HitPathCP path,  BPBaseButtonEventCP ev) override;
	//virtual void   _OnReceive(Utf8CP messageType, JsonValueCR messageDataObj) override;
	virtual bool   _onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown) override;
	virtual UsesDragSelect      _allowDragSelect() override { return USES_DRAGSELECT_Box; }
	virtual bool                _needPointForSelection()   override { return false; }
	virtual bool                _needPointForDynamics()  override { return false; }
	virtual bool                _needAcceptPoint()override { return true; }
	virtual bool                _wantAdditionalLocate( BPBaseButtonEventCP ev)override { return true; }
	virtual bool                _onModifierKeyTransition(bool wentDown, int key)override;
	//virtual p3d::StatusInt          _onEntityModify(BPEntityR el) override;

	virtual BPEntityPtr _buildLocateAgenda(BPPickDataCP path, BPBaseButtonEventCP ev) override;

private:
	vector<BPEntityPtr> m_vcEEH;
	vector<GePoint3d>   m_vctPoints;
	vector<GePoint3d>   m_vctRawPoints;
};
