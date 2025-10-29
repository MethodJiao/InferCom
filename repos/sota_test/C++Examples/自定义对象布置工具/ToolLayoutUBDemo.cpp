#include "stdafx.h"
#include "ToolLayoutUBDemo.h"

using namespace DemoObject;

ToolLayoutUBDemo::ToolLayoutUBDemo()
{
	m_ptrUB = UniversalBeamDemo::create();
}


ToolLayoutUBDemo::~ToolLayoutUBDemo()
{

}

void ToolLayoutUBDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
}

void ToolLayoutUBDemo::_onRestartTool()
{
	ToolLayoutUBDemo* newTool = new ToolLayoutUBDemo();
	newTool->installTool();
}

bool ToolLayoutUBDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	BIMBase::Core::BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == nullptr)
		return false;

	BPModelP pModel = ev->getViewport()->getTargetModel();
	if (pModel == nullptr)
		return false;

	if (m_ptrUB.isNull())
		return false;

	GePoint3d ptDynamic = *ev->getPoint();
	ptDynamic.z = 0;//屏幕点击点Z向高度设置为楼层标高
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	BPPlacement placementNew = m_ptrUB->getPlacement();
	placementNew.setOrigin(ptDynamic);

	//设置基本信息
	m_ptrUB->setPlacement(placementNew);


	auto rt = m_ptrUB->getDataKey();

	//增加扩展属性
	BPExtendPropertySet extendPropertySet = m_ptrUB->getExtendPropertySet();
	BPPropertyValue valueExtend;
	valueExtend.m_value.m_valueString = _T("工字钢自定义属性");
	valueExtend.m_type = BPPropertyValue::String;
	extendPropertySet.setSubParam(_T("扩展属性"), valueExtend);
	m_ptrUB->setExtendPropertySet(extendPropertySet);

	//增加构件到工程中
	if (SUCCESS != m_ptrUB->addToProject(*pProject, pModel->getModelId()))
	{
		AfxMessageBox(L"Can not add to project!");
		return false;
	}

	//获取当前视图，强制刷新界面
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort) return false;
	if (pViewPort)
	{
		pViewPort->updateView();
	}
	return true;
}


bool ToolLayoutUBDemo::_onResetButton(BPBaseButtonEventCP)
{
	_exitTool();
	return true;
}

void ToolLayoutUBDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	m_ptrUB = UniversalBeamDemo::create();
	if (NULL == ev)
		return;

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == nullptr)
		return;
	BPModelP pModel = ev->getViewport()->getTargetModel();
	if (pModel == nullptr)
		return;

	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();
	//获取鼠标屏幕点击的点
	GePoint3d ptDynamic = *ev->getPoint();
	ptDynamic.z = 0;
	BPPlacement placementNew = m_ptrUB->getPlacement();
	placementNew.setOrigin(ptDynamic);
	m_ptrUB->setPlacement(placementNew);

	//根据构造的墙数据随着鼠标移动动态显示墙构件
	BPGraphicsPtr ptrGraphics;
	ptrGraphics = m_ptrUB->createPhysicalGraphics(*pProject, curModelId, true);

	if (!ptrGraphics.isValid())
		return;

	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, m_ptrUB->getPlacement().toTransform());
	ptrGraphics->finish();

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphics->getEntityR());
}

bool ToolLayoutUBDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

//对工具进行注册
BPTool* CreateUBTool()
{
	ToolLayoutUBDemo* tool = new ToolLayoutUBDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutUBDemo", &CreateUBTool);
AutoDoRegisterFunctionsEnd