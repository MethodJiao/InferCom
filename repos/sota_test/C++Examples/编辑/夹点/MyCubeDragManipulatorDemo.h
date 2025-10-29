#pragma once
/** @class  
 *  @brief   立方体夹点设置
 *  @author  北京构力
 *  ------------------------------------------------------------
 *  版本历史       注释                       日期
 *  ------------------------------------------------------------
 *  @version v1.0  初始版本              2020/4/27
 *  ------------------------------------------------------------
 *  @note:  -  
 */
class MyCubeDragManipulatorDemo : public BPPointDragManipulator
{
	DefineSuper(BPPointDragManipulator)

public:
	static MyCubeDragManipulatorDemo* Create();
protected:
	MyCubeDragManipulatorDemo();
	~MyCubeDragManipulatorDemo();

protected:
	//增加夹点控制点
	virtual bool _onCreateControls(::BIMBase::Core::BPEntityCR) override;
	//拖拽移动夹点
	virtual StatusInt _doDragControls(BPEntityR elHandle, BPBaseButtonEventCR ev, bool isDynamics) override;
	//选中构件时临时绘制
	virtual void _onDraw(BPEntityCR, BPViewportP) override;
	
private:
	int addDragControlArrow(GePoint3dCR point, GeVec3dCR vecDirection, GeVec3dCR vecSide);
	int addControlArrowPoint(GePoint3dCR point, GeVec3dCR vecDir);
	BPGraphicsPtr createDimensionLinear(DemoObject::CubeDemoP cube, BPProjectR project, BPViewportP viewport);
};

//自绘三角夹点
struct DragRectTriangle : public BPPointDragManipulator::ControlPoint
{
	GeVec3d m_direction;
	GeVec3d m_rotate;
	int m_nLength;
	int m_menuColor;
	int m_transparency;
	DragRectTriangle(GeVec3dCR direction, GeVec3dCR rotate, GePoint3dCR point, int nlength = 10, int menuColor = 2, int transparency = 150);
	virtual void _draw(BPViewportR vp, GeTransformCP GeTransform = NULL) const override;
	virtual bool _locate(::BIMBase::Core::BPBaseButtonEventCR ev, p3d::GeTransformCP transform = NULL) const override;
	//求视图的眼睛射线
	static void InitBoresite(GeRay3dR boresite, GePoint3dCR spacePoint, GeMatrix4dCR worldToLocal);
};

class MyCubeDragManipulatorDemoExtension : public ::BIMBase::Data::IBPDragManipulatorExtension
{
protected:
	virtual BPIDragManipulatorP _getIDragManipulator(::BIMBase::Core::BPEntityCR elHandle, ::BIMBase::Core::BPPickDataCP path) override;
};