#pragma once

class ToolBlend : public BPEntitySelectSetTool
{
	DefineSuper(BPEntitySelectSetTool)
public:
	ToolBlend();
	~ToolBlend();
	enum LineType
	{
		Straight,
		Bspline,
	};

protected:
	virtual Utf8CP _getToolName() const override { return "blendDemo"; }
	virtual bool   _onDataButton( BPBaseButtonEventCP) override;
	virtual void   _onRestartTool() override;
	virtual bool   _onResetButton( BPBaseButtonEventCP) override;
	virtual void   _onDynamicFrame( BPBaseButtonEventCP) override;
	virtual void   _onPostInstall() override;
	virtual ::BIMBase::Core::BPEntityPtr _buildLocateAgenda(BIMBase::Core::BPPickDataCP path, BPBaseButtonEventCP ev) override;
	virtual bool   _onKeyTransition(bool bWentDown, ::p3d::platform::P3DVirtualKey key, bool bShiftIsDown, bool bCtrlIsDown) override;
	virtual UsesDragSelect      _allowDragSelect() override { return USES_DRAGSELECT_Box; }
	virtual bool                _needPointForSelection()   override { return false; }
	virtual bool                _needPointForDynamics()  override { return false; }
	virtual bool                _needAcceptPoint()override { return true; }
	virtual bool                _wantAdditionalLocate( BPBaseButtonEventCP ev)override { return true; }
	virtual bool                _onModifierKeyTransition(bool bWentDown, int key)override;

private:
	LineType  m_eLineType;
	vector<::BIMBase::Core::BPEntityPtr> m_vcEEH;
	pvector<GeCurveArrayPtr>  m_vctCurve;
	::BIMBase::Core::BPEntityPtr             m_ptrInitEntity;
	bool createBlend(BPGraphicsPtr ptrGraphic, pvector<GeCurveArrayPtr> vctCurves);
};
