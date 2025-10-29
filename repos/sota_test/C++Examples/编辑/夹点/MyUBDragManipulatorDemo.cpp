#include "stdafx.h"
#include "MyUBDragManipulatorDemo.h"
#include "UniversalBeamDemo.h"
#include "adsarxfunc\ADSARXFUN.h"

using namespace DemoObject;

MyUBDragManipulatorDemo::MyUBDragManipulatorDemo()
{
}


MyUBDragManipulatorDemo::~MyUBDragManipulatorDemo()
{
}

bool MyUBDragManipulatorDemo::_onCreateControls(::BIMBase::Core::BPEntityCR eh)
{
	//根据传入的BPEntity信息获取对象实例
	BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(eh);
	if (!ptrData.isValid())
		return false;

	//根据根据实例初始化
	UniversalBeamDemo pbUB;
	pbUB.initFromData(*ptrData);

	GePoint3d startPt = pbUB.getStartPoint();
	GePoint3d endPt = pbUB.getEndPoint();
	GePoint3d midPt = pbUB.getCenterPoint();
	GePoint3d upStartPt = pbUB.getUpStartPoint();
	GePoint3d upEndtPt = pbUB.getUpEndPoint();

	//原默认夹点
	addControlPoint(startPt);
	addControlPoint(midPt);
	addControlPoint(endPt);
	return true;
}

StatusInt MyUBDragManipulatorDemo::_doDragControls(BPEntityR elHandle, BPBaseButtonEventCR ev, bool isDynamics)
{
	//根据传入的BPEntity信息获取对象实例
	BPDataPtr ptrData = BPDataUtil::getDataOnEntity(elHandle);
	if (!ptrData.isValid())
		return false;

	//根据实例初始化
	UniversalBeamDemo pbUB;
	pbUB.initFromData(*ptrData);

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

	if (index == 1)
	{
		//根据鼠标移动点与原始点求转换矩阵
		GePoint3d offset = { cursorPt.x - m_orginPt.x, cursorPt.y - m_orginPt.y, cursorPt.z - m_orginPt.z };
		GeTransform tm = GeTransform::create(offset);
		pbUB.onTransform(tm);
	}
	else if (index == 0)
	{
		GeTransform curTm = pbUB.getPlacement().toTransform();
		GePoint3d ptOri{};
		GeVec3d vecX, vecY, vecZ;
		curTm.getOriginAndVectors(ptOri, vecX, vecY, vecZ);
		GePoint3d ptOriNew = ptOri;
		ptOri -= vecY * pbUB.getWidth() / 2.0;
		double dLenNew = pbUB.getLength() / 2.0;
		GeVec3d vecXNew = vecX;
		//根据新的终点重新计算长度及坐标系
		dLenNew = cursorPt.distance(ptOri);
		vecXNew = GeVec3d::createByStartEndNormalize(ptOri, cursorPt);
		vecXNew *= -1;

		GeVec3d vecYNew = vecZ ^ vecXNew;
		//根据新计算的坐标及原点得到转换矩阵
		GeTransform transNew = GeTransform::createByOriginAndVectors(ptOriNew, vecXNew, vecYNew, vecZ);
		BPPlacement placNew;
		placNew.fromTransform(transNew);
		//墙设置新的转换矩阵，转换到新的位置
		pbUB.setPlacement(placNew);
		pbUB.setLength(dLenNew * 2.0);
	}
	else if (index == 2)
	{
		GeTransform curTm = pbUB.getPlacement().toTransform();
		GePoint3d ptOri{};
		GeVec3d vecX, vecY, vecZ;
		curTm.getOriginAndVectors(ptOri, vecX, vecY, vecZ);
		GePoint3d ptOriNew = ptOri;
		ptOri -= vecY * pbUB.getWidth() / 2.0;
		double dLenNew = pbUB.getLength() / 2.0;
		GeVec3d vecXNew = vecX;
		//根据新的终点重新计算长度及坐标系
		dLenNew = cursorPt.distance(ptOri);
		vecXNew = GeVec3d::createByStartEndNormalize(ptOri, cursorPt);

		GeVec3d vecYNew = vecZ ^ vecXNew;
		//根据新计算的坐标及原点得到转换矩阵
		GeTransform transNew = GeTransform::createByOriginAndVectors(ptOriNew, vecXNew, vecYNew, vecZ);
		BPPlacement placNew;
		placNew.fromTransform(transNew);
		//墙设置新的转换矩阵，转换到新的位置
		pbUB.setPlacement(placNew);
		pbUB.setLength(dLenNew * 2.0);
	}
	::BIMBase::PModelId curModelId = ev.getViewport()->getTargetModel()->getModelId();

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return ERROR;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == NULL)
		return ERROR;

	if (isDynamics)
	{//动态显示
		BPGraphicsPtr  ptrGraphics = pbUB.createPhysicalGraphics(*pProject, curModelId, true);
		if (ptrGraphics.isValid())
		{
			BIMBase::Data::BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, pbUB.getPlacement().toTransform());
			elHandle = ptrGraphics->getEntityR();
		}
	}
	else
	{//最终点击后Replace
		return pbUB.replaceInProject(*pProject);
	}

	return SUCCESS;
}

MyUBDragManipulatorDemo* MyUBDragManipulatorDemo::Create()
{
	return new MyUBDragManipulatorDemo();
}

BPIDragManipulatorP MyUBDragManipulatorDemoExtension::_getIDragManipulator(::BIMBase::Core::BPEntityCR elHandle, ::BIMBase::Core::BPPickDataCP path)
{
	return MyUBDragManipulatorDemo::Create();
}
