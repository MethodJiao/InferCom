#pragma once
class InsulatorDragManipulatorDemo : public BPPointDragManipulator
{
	DefineSuper(BPPointDragManipulator)

public:
	static InsulatorDragManipulatorDemo* Create();
protected:
	InsulatorDragManipulatorDemo();
	~InsulatorDragManipulatorDemo();

protected:
	//增加夹点控制点
	virtual bool _onCreateControls(::BIMBase::Core::BPEntityCR) override;
	//拖拽移动夹点
	virtual StatusInt _doDragControls(BPEntityR elHandle, BPBaseButtonEventCR ev, bool isDynamics) override;

};
class InsulatorDragManipulatorDemoExtension : public ::BIMBase::Data::IBPDragManipulatorExtension
{
protected:
	virtual BPIDragManipulatorP _getIDragManipulator(::BIMBase::Core::BPEntityCR elHandle, ::BIMBase::Core::BPPickDataCP path) override;
};
