#include "stdafx.h"
#include "InsulatorDragManipulatorDemo.h"
#include "InsulatorDemo.h"
#include "adsarxfunc\ADSARXFUN.h"

#define msGeomConst_piOver2     1.57079632679489660000e+000

//USING_NAMESPACE_PBBIM_DIM
using namespace DemoObject;


InsulatorDragManipulatorDemo::InsulatorDragManipulatorDemo()
{
}


InsulatorDragManipulatorDemo::~InsulatorDragManipulatorDemo()
{
}

bool InsulatorDragManipulatorDemo::_onCreateControls(::BIMBase::Core::BPEntityCR eh)
{
	//根据传入的BPEntity信息获取对象实例
	BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(eh);
	if (!ptrData.isValid())
		return false;

	//根据根据实例初始化
	InsulatorDemo pbInsulator;
	pbInsulator.initFromData(*ptrData);
	GePoint3d ptCenter = pbInsulator.getCenter();
	//原默认夹点
	addControlPoint(ptCenter);

	return true;
}

StatusInt InsulatorDragManipulatorDemo::_doDragControls(BPEntityR elHandle, BPBaseButtonEventCR ev, bool isDynamics)
{
	//根据传入的BPEntity信息获取对象实例
	BPDataPtr ptrData = BPDataUtil::getDataOnEntity(elHandle);
	if (!ptrData.isValid())
		return false;

	//根据实例初始化
	InsulatorDemo pbInsulator;
	pbInsulator.initFromData(*ptrData);

	//获取点击夹点后鼠标点
	GePoint3d cursorPt = GePoint3d::create(0, 0, 0);
	_getAdjustedPoint(cursorPt, ev);


	//获取点击夹点的编号，编号根据加入夹点的顺序确定
	GePoint3d m_orginPt;
	SIZE_T index = -1;
	for (SIZE_T i = 0; i < m_controls.m_locations.size(); i++)
	{
		if (CONTROL_STATE_Flashed == m_controls.m_locations[i]->m_state)
		{
			m_orginPt = m_controls.m_locations[i]->m_point;
			index = i;
			break;
		}
	}
	if (index < 0)
		return ERROR;



	cursorPt.z = m_orginPt.z;
	//平移
	//通过转换矩阵获取原点及坐标信息
	GePoint3d offset = { cursorPt.x - m_orginPt.x, cursorPt.y - m_orginPt.y, cursorPt.z - m_orginPt.z };
	GeTransform tm = GeTransform::create(offset);
	pbInsulator.setCenter(cursorPt);


	::BIMBase::PModelId curModelId = ev.getViewport()->getTargetModel()->getModelId();

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return ERROR;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == NULL)
		return ERROR;

	if (isDynamics)
	{//动态显示
		BPGraphicsPtr  ptrGraphics = pbInsulator.createPhysicalGraphics(*pProject, curModelId, true);
		if (ptrGraphics.isValid())
		{
			BIMBase::Data::BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, pbInsulator.getPlacement().toTransform());
			elHandle = ptrGraphics->getEntityR();
		}
	}
	else
	{//最终点击后Replace
		return pbInsulator.replaceInProject(*pProject);
	}

	return SUCCESS;
}



InsulatorDragManipulatorDemo* InsulatorDragManipulatorDemo::Create()
{
	return new InsulatorDragManipulatorDemo();
}

BPIDragManipulatorP InsulatorDragManipulatorDemoExtension::_getIDragManipulator(::BIMBase::Core::BPEntityCR elHandle, ::BIMBase::Core::BPPickDataCP path)
{
	return InsulatorDragManipulatorDemo::Create();
}
