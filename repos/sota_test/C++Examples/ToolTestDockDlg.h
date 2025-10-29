#pragma once

class ToolTestDockDlg : public BIMBase::Core::BPEntitySelectSetTool//, public PBBim::PBUIToWorkMessageEvent
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolTestDockDlg();
	~ToolTestDockDlg();

protected:
	//工具ID
	virtual Utf8CP _getToolName() const override { return "TestDockDlg"; }

	//virtual bool   _OnInstall() override;

	//工具启动后响应函数
	virtual void _onPostInstall() override;

	//重启工具响应函数
	virtual void _onRestartTool() override;

	virtual void   _setupAndPromptForNextAction() override;

	//点击鼠标左键响应函数
	virtual bool   _onDataButton(BPBaseButtonEventCP) override;
	
	//点击鼠标右键响应函数
	virtual bool	_onResetButton(BPBaseButtonEventCP) override;

	//virtual void    _OnReceive(Utf8CP messageType, JsonValueCR messageDataObj) override;

	virtual UsesDragSelect      _allowDragSelect() override                { return USES_DRAGSELECT_Box; }
	virtual bool                _needPointForSelection()   override        { return false; }
	virtual bool                _needPointForDynamics()  override          { return false; }
	virtual bool                _needAcceptPoint()override                 { return true; }
	virtual bool                _wantAdditionalLocate(BPBaseButtonEventCP ev)override { return true; }

	//virtual StatusInt			_onEntityModify(BPEntityR el) override;
	virtual bool                _onModifierKeyTransition(bool wentDown, int key) override;

	virtual BPEntityPtr _buildLocateAgenda(BPPickDataCP path, BPBaseButtonEventCP ev) override;

	virtual void _exitTool();
	
private:
	vector<BPEntityPtr> m_vcEEH;
	
};

