#pragma once
/** @class
 *  @brief   工字钢夹点设置
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/27
 *  ------------------------------------------------------------
 *  @note:  -
 */
class MyUBDragManipulatorDemo : public BPPointDragManipulator
{
	DefineSuper(BPPointDragManipulator)

public:
	static MyUBDragManipulatorDemo* Create();
protected:
	MyUBDragManipulatorDemo();
	~MyUBDragManipulatorDemo();

protected:
	//增加夹点控制点
	virtual bool _onCreateControls(::BIMBase::Core::BPEntityCR) override;
	//拖拽移动夹点
	virtual StatusInt _doDragControls(BPEntityR elHandle, BPBaseButtonEventCR ev, bool isDynamics) override;
};

class MyUBDragManipulatorDemoExtension : public ::BIMBase::Data::IBPDragManipulatorExtension
{
protected:
	virtual BPIDragManipulatorP _getIDragManipulator(::BIMBase::Core::BPEntityCR elHandle, ::BIMBase::Core::BPPickDataCP path) override;
};